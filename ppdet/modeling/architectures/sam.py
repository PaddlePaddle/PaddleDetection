# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['SAM']


@register
class SAM(BaseArch):
    __category__ = 'architecture'
    mask_threshold = 0.0

    def __init__(self,
                 image_encoder='ImageEncoderViT',
                 prompt_encoder='PromptEncoder',
                 mask_decoder='MaskDecoder'):
        super(SAM, self).__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # image_encoder
        image_encoder = create(cfg['image_encoder'])

        # prompt_encoder
        prompt_encoder = create(cfg['prompt_encoder'])

        # mask_decoder
        mask_decoder = create(cfg['mask_decoder'])

        return {
            'image_encoder': image_encoder,
            'prompt_encoder': prompt_encoder,
            "mask_decoder": mask_decoder,
        }

    def _forward(self):
        if self.training:
            raise ValueError

        curr_embedding = self.image_encoder(self.inputs)

        outputs = []
        # for image_record, curr_embedding in zip(batched_input, image_embeddings):
        if 1:
            image_record = self.inputs
            multimask_output = self.inputs

            if "point_coords" in image_record:
                points = (image_record["point_coords"],
                          image_record["point_labels"])
            else:
                points = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None), )

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output, )

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"], )
            masks = masks > self.mask_threshold
            outputs.append({
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            })
        return outputs[0]

    def postprocess_masks(self, masks, input_size, original_size):
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False, )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
