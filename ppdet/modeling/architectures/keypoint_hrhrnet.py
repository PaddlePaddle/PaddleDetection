# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.optimize import linear_sum_assignment
from collections import abc, defaultdict
import numpy as np
import paddle

from ppdet.core.workspace import register, create, serializable
from .meta_arch import BaseArch
from .. import layers as L
from ..keypoint_utils import transpred

__all__ = ['HigherHrnet']


@register
class HigherHrnet(BaseArch):
    __category__ = 'architecture'

    def __init__(self,
                 backbone='Hrnet',
                 hrhrnet_head='HigherHrnetHead',
                 post_process='HrHrnetPostProcess',
                 eval_flip=True,
                 flip_perm=None):
        """
        HigherHrnet network, see https://arxiv.org/abs/

        Args:
            backbone (nn.Layer): backbone instance
            hrhrnet_head (nn.Layer): keypoint_head instance
            bbox_post_process (object): `BBoxPostProcess` instance
        """
        super(HigherHrnet, self).__init__()
        self.backbone = backbone
        self.hrhrnet_head = hrhrnet_head
        self.post_process = HrHrnetPostProcess()
        self.flip = eval_flip
        self.flip_perm = paddle.to_tensor(flip_perm)
        self.deploy = False

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # head
        kwargs = {'input_shape': backbone.out_shape}
        hrhrnet_head = create(cfg['hrhrnet_head'], **kwargs)
        post_process = create(cfg['post_process'])

        return {
            'backbone': backbone,
            "hrhrnet_head": hrhrnet_head,
            "post_process": post_process,
        }

    def _forward(self):
        batchsize = self.inputs['image'].shape[0]
        if self.flip and not self.training and not self.deploy:
            self.inputs['image'] = paddle.concat(
                (self.inputs['image'], paddle.flip(self.inputs['image'], [3])))
        body_feats = self.backbone(self.inputs)

        if self.training:
            return self.hrhrnet_head(body_feats, self.inputs)
        else:
            outputs = self.hrhrnet_head(body_feats)
            if self.deploy:
                return outputs, [1]
            if self.flip:
                outputs = [paddle.split(o, 2) for o in outputs]
                output_rflip = [
                    paddle.flip(paddle.gather(o[1], self.flip_perm, 1), [3])
                    for o in outputs
                ]
                output1 = [o[0] for o in outputs]
                heatmap = (output1[0] + output_rflip[0]) / 2.
                tagmaps = [output1[1], output_rflip[1]]
                outputs = [heatmap] + tagmaps

            res_lst = []
            bboxnums = []
            for idx in range(batchsize):
                item = [o[idx:(idx + 1)] for o in outputs]

                h = self.inputs['im_shape'][idx, 0].numpy().item()
                w = self.inputs['im_shape'][idx, 1].numpy().item()
                kpts, scores = self.post_process(item, h, w)
                res_lst.append([kpts, scores])
                bboxnums.append(1)

            return res_lst, bboxnums

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        outputs = {}
        res_lst, bboxnums = self._forward()
        outputs['keypoint'] = res_lst
        outputs['bbox_num'] = bboxnums
        return outputs


@register
@serializable
class HrHrnetPostProcess(object):
    def __init__(self, max_num_people=30, heat_thresh=0.2, tag_thresh=1.):
        self.interpolate = L.Upsample(2, mode='bilinear')
        self.pool = L.MaxPool(5, 1, 2)
        self.max_num_people = max_num_people
        self.heat_thresh = heat_thresh
        self.tag_thresh = tag_thresh

    def lerp(self, j, y, x, heatmap):
        H, W = heatmap.shape[-2:]
        left = np.clip(x - 1, 0, W - 1)
        right = np.clip(x + 1, 0, W - 1)
        up = np.clip(y - 1, 0, H - 1)
        down = np.clip(y + 1, 0, H - 1)
        offset_y = np.where(heatmap[j, down, x] > heatmap[j, up, x], 0.25,
                            -0.25)
        offset_x = np.where(heatmap[j, y, right] > heatmap[j, y, left], 0.25,
                            -0.25)
        return offset_y + 0.5, offset_x + 0.5

    def __call__(self, inputs, original_height, original_width):

        # resize to image size
        inputs = [self.interpolate(x) for x in inputs]
        # aggregate
        heatmap = inputs[0]
        if len(inputs) == 3:
            tagmap = paddle.concat(
                (inputs[1].unsqueeze(4), inputs[2].unsqueeze(4)), axis=4)
        else:
            tagmap = inputs[1].unsqueeze(4)

        N, J, H, W = heatmap.shape
        assert N == 1, "only support batch size 1"
        # topk
        maximum = self.pool(heatmap)
        maxmap = heatmap * (heatmap == maximum)
        maxmap = maxmap.reshape([N, J, -1])
        heat_k, inds_k = maxmap.topk(self.max_num_people, axis=2)
        heatmap = heatmap[0].cpu().detach().numpy()
        tagmap = tagmap[0].cpu().detach().numpy()
        heats = heat_k[0].cpu().detach().numpy()
        inds_np = inds_k[0].cpu().detach().numpy()
        y = inds_np // W
        x = inds_np % W
        tags = tagmap[np.arange(J)[None, :].repeat(self.max_num_people),
                      y.flatten(), x.flatten()].reshape(J, -1, tagmap.shape[-1])
        coords = np.stack((y, x), axis=2)
        # threshold
        mask = heats > self.heat_thresh
        # cluster
        cluster = defaultdict(lambda: {
            'coords': np.zeros((J, 2), dtype=np.float32),
            'scores': np.zeros(J, dtype=np.float32),
            'tags': []
        })
        for jid, m in enumerate(mask):
            num_valid = m.sum()
            if num_valid == 0:
                continue
            valid_inds = np.where(m)[0]
            valid_tags = tags[jid, m, :]
            if len(cluster) == 0:  # initialize
                for i in valid_inds:
                    tag = tags[jid, i]
                    key = tag[0]
                    cluster[key]['tags'].append(tag)
                    cluster[key]['scores'][jid] = heats[jid, i]
                    cluster[key]['coords'][jid] = coords[jid, i]
                continue
            candidates = list(cluster.keys())[:self.max_num_people]
            centroids = [
                np.mean(
                    cluster[k]['tags'], axis=0) for k in candidates
            ]
            num_clusters = len(centroids)
            # shape is (num_valid, num_clusters, tag_dim)
            dist = valid_tags[:, None, :] - np.array(centroids)[None, ...]
            l2_dist = np.linalg.norm(dist, ord=2, axis=2)
            # modulate dist with heat value, see `use_detection_val`
            cost = np.round(l2_dist) * 100 - heats[jid, m, None]
            # pad the cost matrix, otherwise new pose are ignored
            if num_valid > num_clusters:
                cost = np.pad(cost, ((0, 0), (0, num_valid - num_clusters)),
                              constant_values=((0, 0), (0, 1e-10)))
            rows, cols = linear_sum_assignment(cost)
            for y, x in zip(rows, cols):
                tag = tags[jid, y]
                if y < num_valid and x < num_clusters and \
                   l2_dist[y, x] < self.tag_thresh:
                    key = candidates[x]  # merge to cluster
                else:
                    key = tag[0]  # initialize new cluster
                cluster[key]['tags'].append(tag)
                cluster[key]['scores'][jid] = heats[jid, y]
                cluster[key]['coords'][jid] = coords[jid, y]

        # shape is [k, J, 2] and [k, J]
        pose_tags = np.array([cluster[k]['tags'] for k in cluster])
        pose_coords = np.array([cluster[k]['coords'] for k in cluster])
        pose_scores = np.array([cluster[k]['scores'] for k in cluster])
        valid = pose_scores > 0

        pose_kpts = np.zeros((pose_scores.shape[0], J, 3), dtype=np.float32)
        if valid.sum() == 0:
            return pose_kpts, pose_kpts

        # refine coords
        valid_coords = pose_coords[valid].astype(np.int32)
        y = valid_coords[..., 0].flatten()
        x = valid_coords[..., 1].flatten()
        _, j = np.nonzero(valid)
        offsets = self.lerp(j, y, x, heatmap)
        pose_coords[valid, 0] += offsets[0]
        pose_coords[valid, 1] += offsets[1]

        # mean score before salvage
        mean_score = pose_scores.mean(axis=1)
        pose_kpts[valid, 2] = pose_scores[valid]

        # TODO can we remove the outermost loop altogether
        # salvage missing joints

        if True:
            for pid, coords in enumerate(pose_coords):
                # vj = np.nonzero(valid[pid])[0]
                # vyx = coords[valid[pid]].astype(np.int32)
                # tag_mean = tagmap[vj, vyx[:, 0], vyx[:, 1]].mean(axis=0)

                tag_mean = np.array(pose_tags[pid]).mean(
                    axis=0)  #TODO: replace tagmap sample by history record

                norm = np.sum((tagmap - tag_mean)**2, axis=3)**0.5
                score = heatmap - np.round(norm)  # (J, H, W)
                flat_score = score.reshape(J, -1)
                max_inds = np.argmax(flat_score, axis=1)
                max_scores = np.max(flat_score, axis=1)
                salvage_joints = (pose_scores[pid] == 0) & (max_scores > 0)
                if salvage_joints.sum() == 0:
                    continue
                y = max_inds[salvage_joints] // W
                x = max_inds[salvage_joints] % W
                offsets = self.lerp(salvage_joints.nonzero()[0], y, x, heatmap)
                y = y.astype(np.float32) + offsets[0]
                x = x.astype(np.float32) + offsets[1]
                pose_coords[pid][salvage_joints, 0] = y
                pose_coords[pid][salvage_joints, 1] = x
                pose_kpts[pid][salvage_joints, 2] = max_scores[salvage_joints]
        pose_kpts[..., :2] = transpred(pose_coords[..., :2][..., ::-1],
                                       original_height, original_width,
                                       min(H, W))
        return pose_kpts, mean_score
