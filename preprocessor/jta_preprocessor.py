import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

from preprocessor.preprocessor import Processor, OUTPUT_DIR


class JTAPreprocessor(Processor):
    def __init__(self, is_3d, mask, dataset_path, is_disentangle, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name, is_interactive):
        super(JTAPreprocessor, self).__init__(dataset_path, is_disentangle, obs_frame_num, pred_frame_num,
                                              skip_frame_num, use_video_once, custom_name, is_interactive)
        self.dataset_total_frame_num = 900
        self.is_3d = is_3d
        self.mask = mask
        if is_3d:
            self.start_dim = 5
            self.end_dim = 8
        else:
            self.start_dim = 3
            self.end_dim = 5
        if self.is_3d:
            if self.is_interactive:
                self.output_dir = os.path.join(OUTPUT_DIR, 'JTA_interactive', '3D')
            else:
                self.output_dir = os.path.join(OUTPUT_DIR, 'JTA', '3D')
        else:
            if self.is_interactive:
                self.output_dir = os.path.join(OUTPUT_DIR, 'JTA_interactive', '2D')
            else:
                self.output_dir = os.path.join(OUTPUT_DIR, 'JTA', '2D')

    def normal(self, data_type='train'):
        print('start creating JTA normal static data ... ')
        header = ['video_section', 'observed_pose', 'future_pose', 'observed_mask', 'future_mask']
        if self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.csv'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_JTA.csv'
        with open(os.path.join(self.output_dir, output_file_name), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)

        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.dataset_total_frame_num // (
                total_frame_num * self.skip_frame_num) if self.use_video_once is False else 1

        for entry in os.scandir(self.dataset_path):
            if not entry.path.endswith('.json'):
                continue
            with open(entry.path, 'r') as json_file:
                print(f'file name: {entry.name}')
                video_number = re.search('seq_(\d+)', entry.name).group(1)
                data = []
                matrix = json.load(json_file)
                matrix = np.array(matrix)
                for i in range(section_range):
                    video_data = {
                        'obs_pose': defaultdict(list),
                        'future_pose': defaultdict(list),
                        'obs_mask': defaultdict(list),
                        'future_mask': defaultdict(list)
                    }
                    obs = []
                    obs_mask = []
                    future = []
                    future_mask = []
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        frame_data = {
                            'pose': defaultdict(list),
                            'mask': defaultdict(list)
                        }
                        frame = matrix[matrix[:, 0] == i * total_frame_num * self.skip_frame_num + j]  # find frame data
                        for pose in frame:
                            masked = 0
                            # pose data
                            for kp_position in range(self.start_dim, self.end_dim):
                                frame_data['pose'][pose[1]].append(pose[kp_position])
                            # mask data
                            for masking_state in range(8, 10):
                                masked += pose[masking_state]
                            frame_data['mask'][pose[1]].append(1 if masked > 0 else 0)
                        for p_id in frame_data['pose'].keys():
                            if j <= self.obs_frame_num * self.skip_frame_num:
                                video_data['obs_pose'][p_id].append(frame_data['pose'][p_id])
                                video_data['obs_mask'][p_id].append(frame_data['mask'][p_id])
                            else:
                                video_data['future_pose'][p_id].append(frame_data['pose'][p_id])
                                video_data['future_mask'][p_id].append(frame_data['mask'][p_id])
                    for p_id in video_data['obs_pose']:
                        if p_id in video_data['future_pose'].keys() and video_data['obs_pose'][
                            p_id].__len__() == self.obs_frame_num and \
                                video_data['future_pose'][
                                    p_id].__len__() == self.pred_frame_num:
                            obs.append(video_data['obs_pose'][p_id])
                            obs_mask.append(video_data['obs_mask'][p_id])
                            future.append(video_data['future_pose'][p_id])
                            future_mask.append(video_data['future_mask'][p_id])
                    if self.is_interactive:
                        for p_id in range(len(obs)):
                            data.append(['%s-%d' % (video_number, i), obs[p_id], future[p_id], obs_mask[p_id],
                                         future_mask[p_id]])
                    else:
                        data.append(['%s-%d' % (video_number, i), obs, future, obs_mask, future_mask])
                with open(os.path.join(self.output_dir, output_file_name), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data)

    def disentangle(self, data_type='train'):
        print('start creating JTA disentangle static data ...')
        header = ['video_section', 'observed_pose', 'future_pose', 'observed_mask', 'future_mask']
        if self.custom_name:
            global_file_name = f'global_{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_' \
                               f'{self.custom_name}.csv'
            local_file_name = f'local_{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_' \
                              f'{self.custom_name}.csv'
        else:
            global_file_name = f'global_{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_' \
                               f'{self.skip_frame_num}_JTA.csv'
            local_file_name = f'local_{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_' \
                              f'{self.skip_frame_num}_JTA.csv'

        with open(os.path.join(self.output_dir, global_file_name), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        with open(os.path.join(self.output_dir, local_file_name), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)

        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.dataset_total_frame_num // (
                total_frame_num * self.skip_frame_num) if self.use_video_once is False else 1

        for entry in os.scandir(self.dataset_path):
            if not entry.path.endswith('.json'):
                continue
            with open(entry.path, 'r') as json_file:
                print(f'file name: {entry.name}')
                video_number = re.search('seq_(\d+)', entry.name).group(1)
                data_global = []
                data_local = []
                matrix = json.load(json_file)
                matrix = np.array(matrix)
                for i in range(section_range):
                    global_dict = {
                        'obs_pose': defaultdict(list), 'future_pose': defaultdict(list),
                        'obs_mask': defaultdict(list), 'future_mask': defaultdict(list)
                    }
                    local_dict = {
                        'obs_pose': defaultdict(list), 'future_pose': defaultdict(list),
                        'obs_mask': defaultdict(list), 'future_mask': defaultdict(list)
                    }
                    obs_global = []
                    obs_mask_global = []
                    future_global = []
                    future_mask_global = []
                    obs_local = []
                    obs_mask_local = []
                    future_local = []
                    future_mask_local = []
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        frame_global_dict = {
                            'pose': defaultdict(list),
                            'mask': defaultdict(list)
                        }

                        frame_local_dict = {
                            'pose': defaultdict(list),
                            'mask': defaultdict(list)
                        }
                        frame = matrix[matrix[:, 0] == i * total_frame_num * 2 + j]  # find frame data
                        for pose in frame:
                            if pose[2] != 2:
                                continue
                            for kp_position in range(self.start_dim, self.end_dim):
                                frame_global_dict['pose'][pose[1]].append(pose[kp_position])
                            masked = 0
                            for masking_state in range(8, 10):
                                masked += pose[masking_state]
                            frame_global_dict['mask'][pose[1]].append(1 if masked > 0 else 0)
                        for pose in frame:
                            if pose[2] == 2:
                                continue
                            masked = 0
                            for masking_state in range(8, 10):
                                masked += pose[masking_state]
                            frame_local_dict['mask'][pose[1]].append(1 if masked > 0 else 0)
                            temp_local = []
                            for kp_position in range(self.start_dim, self.end_dim):
                                temp_local.append(pose[kp_position])
                            temp_global = np.array(frame_global_dict['pose'][pose[1]])
                            temp_local = np.array(temp_local)
                            frame_local_dict['pose'][pose[1]].extend((temp_local - temp_global).tolist())
                        for p_id in frame_global_dict['pose'].keys():
                            if j <= self.obs_frame_num * self.skip_frame_num:
                                global_dict['obs_pose'][p_id].append(frame_global_dict['pose'][p_id])
                                global_dict['obs_mask'][p_id].append(frame_global_dict['mask'][p_id])
                                local_dict['obs_pose'][p_id].append(frame_local_dict['pose'][p_id])
                                local_dict['obs_mask'][p_id].append(frame_local_dict['mask'][p_id])
                            else:
                                global_dict['future_pose'][p_id].append(frame_global_dict['pose'][p_id])
                                global_dict['future_mask'][p_id].append(frame_global_dict['mask'][p_id])
                                local_dict['future_pose'][p_id].append(frame_local_dict['pose'][p_id])
                                local_dict['future_mask'][p_id].append(frame_local_dict['mask'][p_id])
                    for p_id in global_dict['obs_pose']:
                        if p_id in global_dict['future_pose'].keys() and global_dict['obs_pose'][
                            p_id].__len__() == self.pred_frame_num and \
                                global_dict['future_pose'][
                                    p_id].__len__() == self.pred_frame_num:
                            obs_global.append(global_dict['obs_pose'][p_id])
                            obs_mask_global.append(global_dict['obs_mask'][p_id])
                            future_global.append(global_dict['future_pose'][p_id])
                            future_mask_global.append(global_dict['future_mask'][p_id])

                            obs_local.append(local_dict['obs_pose'][p_id])
                            obs_mask_local.append(local_dict['obs_mask'][p_id])
                            future_local.append(local_dict['future_pose'][p_id])
                            future_mask_local.append(local_dict['future_mask'][p_id])
                    if self.is_interactive:
                        for p_id in range(len(obs_global)):
                            data_global.append(['%s-%d' % (video_number, i), obs_global[p_id], future_global[p_id],
                                                obs_mask_global[p_id], future_mask_global[p_id]])
                            data_local.append(['%s-%d' % (video_number, i), obs_global[p_id], future_global[p_id],
                                               obs_mask_global[p_id], future_mask_global[p_id]])
                    else:
                        data_global.append(
                            ['%s-%d' % (video_number, i), obs_global, future_global, obs_mask_global,
                             future_mask_global])
                        data_local.append(
                            ['%s-%d' % (video_number, i), obs_local, future_local, obs_mask_local, future_mask_local])
                with open(os.path.join(self.output_dir, global_file_name), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data_global)
                with open(os.path.join(self.output_dir, local_file_name), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data_local)
