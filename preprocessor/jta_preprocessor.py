import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

from preprocessor.preprocessor import Processor, OUTPUT_DIR

class JTA_Preprocessor(Processor):
    def __init__(self, is_3d, mask):
        super(JTA_Preprocessor, self).__init__()
        self.frame_num = 900
        self.is_3d = is_3d
        self.mask = mask

    def normal(self, data_type='train'):
        header = ['video_section', 'observed_pose', 'future_pose']
        with open(os.path.join(OUTPUT_DIR, 'JTA_{}.csv'.format(data_type)), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)

        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.frame_num // (total_frame_num * self.skip_frame_num) if self.use_video_once is False else 1
        if self.is_3d:
            start_dim = 5
            end_dim = 8
        else:  # 2D
            start_dim = 3
            end_dim = 5

        for entry in os.scandir(self.dataset_path):
            with open(entry.path, 'r') as json_file:
                print(entry.path)
                video_number = re.search('seq_(\d+)', entry.name).group(1)
                data = []
                matrix = json.load(json_file)
                matrix = np.array(matrix)
                for i in range(section_range):
                    obs_dict = defaultdict(list)
                    future_dict = defaultdict(list)
                    obs = []
                    future = []
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        poses = defaultdict(list)
                        frame = matrix[matrix[:, 0] == i * total_frame_num * self.skip_frame_num + j]  # find frame data
                        for pose in frame:
                            for kp_position in range(start_dim, end_dim):
                                poses[pose[1]].append(pose[kp_position])
                        for p_id in poses.keys():
                            if j <= self.obs_frame_num * self.skip_frame_num:
                                obs_dict[p_id].append(poses[p_id])
                            else:
                                future_dict[p_id].append(poses[p_id])
                    for p_id in obs_dict:
                        if p_id in future_dict.keys() and obs_dict[p_id].__len__() == self.obs_frame_num and \
                                future_dict[
                                    p_id].__len__() == self.pred_frame_num:
                            obs.append(obs_dict[p_id])
                            future.append(future_dict[p_id])
                    data.append(['%s_%d' % (video_number, i), obs, future])
                with open(os.path.join(OUTPUT_DIR, 'JTA_{}.csv'.format(data_type)), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data)

    def mask(self, data_type='train'):
        header = ['video_section', 'observed_mask', 'future_mask']
        with open(os.path.join(OUTPUT_DIR, 'JTA_{}.csv'.format(data_type)), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)

        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.frame_num // (total_frame_num * self.skip_frame_num) if self.use_video_once is False else 1

        for entry in os.scandir(self.dataset_path):
            with open(entry.path, 'r') as json_file:
                print(entry.path)
                video_number = re.search('seq_(\d+)', entry.name).group(1)
                data = []
                matrix = json.load(json_file)
                matrix = np.array(matrix)
                for i in range(section_range):
                    obs_dict = defaultdict(list)
                    future_dict = defaultdict(list)
                    obs = []
                    future = []
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        poses = defaultdict(list)
                        frame = matrix[matrix[:, 0] == i * total_frame_num * self.skip_frame_num + j]  # find frame data
                        for pose in frame:
                            masked = 0
                            for masking_state in range(8, 10):
                                masked += pose[masking_state]
                            poses[pose[1]].append(1 if masked > 0 else 0)
                        for p_id in poses.keys():
                            if j <= self.obs_frame_num * self.skip_frame_num:
                                obs_dict[p_id].append(poses[p_id])
                            else:
                                future_dict[p_id].append(poses[p_id])
                    for p_id in obs_dict:
                        if p_id in future_dict.keys() and obs_dict[p_id].__len__() == self.obs_frame_num and \
                                future_dict[
                                    p_id].__len__() == self.pred_frame_num:
                            obs.append(obs_dict[p_id])
                            future.append(future_dict[p_id])
                    data.append(['%s_%d' % (video_number, i), obs, future])
                with open(os.path.join(OUTPUT_DIR, 'JTA_{}.csv'.format(data_type)), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data)
