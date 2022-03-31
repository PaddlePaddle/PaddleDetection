# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import copy


class Result(object):
    def __init__(self):
        self.res_dict = {
            'det': dict(),
            'mot': dict(),
            'attr': dict(),
            'kpt': dict(),
            'action': dict(),
            'reid': dict()
        }

    def update(self, res, name):
        self.res_dict[name].update(res)

    def get(self, name):
        if name in self.res_dict and len(self.res_dict[name]) > 0:
            return self.res_dict[name]
        return None

    def clear(self, name):
        self.res_dict[name].clear()


class DataCollector(object):
    """
  DataCollector of pphuman Pipeline, collect results in every frames and assign it to each track ids.
  mainly used in mtmct.
  
  data struct:
  collector:
    - [id1]: (all results of N frames)
      - frames(list of int): Nx[int]
      - rects(list of rect): Nx[rect(conf, xmin, ymin, xmax, ymax)]
      - features(list of array(256,)): Nx[array(256,)]
      - qualities(list of float): Nx[float]
      - attrs(list of attr): refer to attrs for details
      - kpts(list of kpts): refer to kpts for details
      - actions(list of actions): refer to actions for details
    ...
    - [idN]
  """

    def __init__(self):
        #id, frame, rect, score, label, attrs, kpts, actions
        self.mots = {
            "frames": [],
            "rects": [],
            "attrs": [],
            "kpts": [],
            "features": [],
            "qualities": [],
            "actions": []
        }
        self.collector = {}

    def append(self, frameid, Result):
        mot_res = Result.get('mot')
        attr_res = Result.get('attr')
        kpt_res = Result.get('kpt')
        action_res = Result.get('action')
        reid_res = Result.get('reid')

        rects = reid_res['rects'] if reid_res is not None else mot_res['boxes']
        for idx, mot_item in enumerate(rects):
            ids = int(mot_item[0])
            if ids not in self.collector:
                self.collector[ids] = copy.deepcopy(self.mots)
            self.collector[ids]["frames"].append(frameid)
            self.collector[ids]["rects"].append([mot_item[2:]])
            if attr_res:
                self.collector[ids]["attrs"].append(attr_res['output'][idx])
            if kpt_res:
                self.collector[ids]["kpts"].append(
                    [kpt_res['keypoint'][0][idx], kpt_res['keypoint'][1][idx]])
            if action_res and (idx + 1) in action_res:
                self.collector[ids]["actions"].append(action_res[idx + 1])
            else:
                # action model generate result per X frames, Not available every frames
                self.collector[ids]["actions"].append(None)
            if reid_res:
                self.collector[ids]["features"].append(reid_res['features'][
                    idx])
                self.collector[ids]["qualities"].append(reid_res['qualities'][
                    idx])

    def get_res(self):
        return self.collector
