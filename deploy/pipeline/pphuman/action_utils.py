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


class KeyPointSequence(object):
    def __init__(self, max_size=100):
        self.frames = 0
        self.kpts = []
        self.bboxes = []
        self.max_size = max_size

    def save(self, kpt, bbox):
        self.kpts.append(kpt)
        self.bboxes.append(bbox)
        self.frames += 1
        if self.frames == self.max_size:
            return True
        return False


class KeyPointBuff(object):
    def __init__(self, max_size=100):
        self.flag_track_interrupt = False
        self.keypoint_saver = dict()
        self.max_size = max_size
        self.id_to_pop = set()
        self.flag_to_pop = False

    def get_state(self):
        return self.flag_to_pop

    def update(self, kpt_res, mot_res):
        kpts = kpt_res.get('keypoint')[0]
        bboxes = kpt_res.get('bbox')
        mot_bboxes = mot_res.get('boxes')
        updated_id = set()

        for idx in range(len(kpts)):
            tracker_id = mot_bboxes[idx, 0]
            updated_id.add(tracker_id)

            kpt_seq = self.keypoint_saver.get(tracker_id,
                                              KeyPointSequence(self.max_size))
            is_full = kpt_seq.save(kpts[idx], bboxes[idx])
            self.keypoint_saver[tracker_id] = kpt_seq

            #Scene1: result should be popped when frames meet max size
            if is_full:
                self.id_to_pop.add(tracker_id)
                self.flag_to_pop = True

        #Scene2: result of a lost tracker should be popped
        interrupted_id = set(self.keypoint_saver.keys()) - updated_id
        if len(interrupted_id) > 0:
            self.flag_to_pop = True
            self.id_to_pop.update(interrupted_id)

    def get_collected_keypoint(self):
        """
            Output (List): List of keypoint results for Skeletonbased Recognition task, where 
                           the format of each element is [tracker_id, KeyPointSequence of tracker_id]
        """
        output = []
        for tracker_id in self.id_to_pop:
            output.append([tracker_id, self.keypoint_saver[tracker_id]])
            del (self.keypoint_saver[tracker_id])
        self.flag_to_pop = False
        self.id_to_pop.clear()
        return output


class ActionVisualHelper(object):
    def __init__(self, frame_life=20):
        self.frame_life = frame_life
        self.action_history = {}

    def get_visualize_ids(self):
        id_detected = self.check_detected()
        return id_detected

    def check_detected(self):
        id_detected = set()
        deperate_id = []
        for mot_id in self.action_history:
            self.action_history[mot_id]["life_remain"] -= 1
            if int(self.action_history[mot_id]["class"]) == 0:
                id_detected.add(mot_id)
            if self.action_history[mot_id]["life_remain"] == 0:
                deperate_id.append(mot_id)
        for mot_id in deperate_id:
            del (self.action_history[mot_id])
        return id_detected

    def update(self, action_res_list):
        for mot_id, action_res in action_res_list:
            if mot_id in self.action_history:
                if int(action_res["class"]) != 0 and int(self.action_history[
                        mot_id]["class"]) == 0:
                    continue
            action_info = self.action_history.get(mot_id, {})
            action_info["class"] = action_res["class"]
            action_info["life_remain"] = self.frame_life
            self.action_history[mot_id] = action_info
