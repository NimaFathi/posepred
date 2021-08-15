import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

from preprocessor.preprocessor import Processor, OUTPUT_DIR


class JTAPreprocessor(Processor):
    def __init__(self, is_3d, mask, is_disentangle, obs_frame_num, pred_frame_num, skip_frame_num, use_video_once):
        super(JTAPreprocessor, self).__init__(is_disentangle, obs_frame_num, pred_frame_num, skip_frame_num,
                                              use_video_once)
        self.dataset_total_frame_num = 900
        self.is_3d = is_3d
        self.mask = mask
        if is_3d:
            self.start_dim = 5
            self.end_dim = 8
        else:
            self.start_dim = 3
            self.end_dim = 5

    def normal(self, data_type='train'):
        header = ['video_section', 'observed_pose', 'future_pose']
        with open(os.path.join(OUTPUT_DIR, 'JTA_{}.csv'.format(data_type)), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)

        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.dataset_total_frame_num // (
                total_frame_num * self.skip_frame_num) if self.use_video_once is False else 1

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
                            for kp_position in range(self.start_dim, self.end_dim):
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

    def disentangle(self, data_type='train'):
        header = ['video_section', 'observed_pose', 'future_pose']
        with open(os.path.join(OUTPUT_DIR, 'JTA_global_{}.csv'.format(data_type)), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        with open(os.path.join(OUTPUT_DIR, 'JTA_local_{}.csv'.format(data_type)), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)

        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.dataset_total_frame_num // (
                total_frame_num * self.skip_frame_num) if self.use_video_once is False else 1

        for entry in os.scandir(self.dataset_path):
            with open(entry.path, 'r') as json_file:
                print(entry.path)
                video_number = re.search('seq_(\d+)', entry.name).group(1)
                data_global = []
                data_local = []
                matrix = json.load(json_file)
                matrix = np.array(matrix)
                for i in range(section_range):
                    print(i)
                    obs_dict_global = defaultdict(list)
                    future_dict_global = defaultdict(list)
                    obs_global = []
                    future_global = []
                    obs_dict_local = defaultdict(list)
                    future_dict_local = defaultdict(list)
                    obs_local = []
                    future_local = []
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        global_poses = defaultdict(list)
                        local_poses = defaultdict(list)
                        frame = matrix[matrix[:, 0] == i * total_frame_num * 2 + j]  # find frame data
                        for pose in frame:
                            if pose[2] != 2:
                                continue
                            for kp_position in range(self.start_dim, self.end_dim):
                                global_poses[pose[1]].append(pose[kp_position])
                        for pose in frame:
                            if pose[2] == 2:
                                continue
                            temp_local = []
                            for kp_position in range(self.start_dim, self.end_dim):
                                temp_local.append(pose[kp_position])
                            temp_global = np.array(global_poses[pose[1]])
                            temp_local = np.array(temp_local)
                            local_poses[pose[1]].extend((temp_local - temp_global).tolist())
                        for p_id in global_poses.keys():
                            if j <= self.obs_frame_num * 2:
                                obs_dict_global[p_id].append(global_poses[p_id])
                                obs_dict_local[p_id].append(local_poses[p_id])
                            else:
                                future_dict_global[p_id].append(global_poses[p_id])
                                future_dict_local[p_id].append(local_poses[p_id])
                    for p_id in obs_dict_global:
                        if p_id in future_dict_global.keys() and obs_dict_global[
                            p_id].__len__() == self.obs_frame_num and \
                                future_dict_global[
                                    p_id].__len__() == self.pred_frame_num:
                            obs_global.append(obs_dict_global[p_id])
                            future_global.append(future_dict_global[p_id])

                            obs_local.append(obs_dict_local[p_id])
                            future_local.append(future_dict_local[p_id])
                    data_global.append(['%s_%d' % (video_number, i), obs_global, future_global])
                    data_local.append(['%s_%d' % (video_number, i), obs_local, future_local])
                with open(os.path.join(OUTPUT_DIR, 'JTA_global_{}.csv'.format(data_type)), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data_global)
                with open(os.path.join(OUTPUT_DIR, 'JTA_local_{}.csv'.format(data_type)), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data_local)

    def mask(self, data_type='train'):
        header = ['video_section', 'observed_mask', 'future_mask']
        with open(os.path.join(OUTPUT_DIR, 'JTA_{}.csv'.format(data_type)), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)

        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.dataset_total_frame_num // (
                total_frame_num * self.skip_frame_num) if self.use_video_once is False else 1

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
                        frame = matrix[matrix[:, 0] == i * total_frame_num * self.skip_frame_num + j]
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