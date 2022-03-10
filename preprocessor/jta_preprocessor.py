import json
import logging
import os
import re
from collections import defaultdict

import jsonlines
import numpy as np

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor

logger = logging.getLogger(__name__)


class JTAPreprocessor(Processor):
    def __init__(self, is_3d, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name):
        super(JTAPreprocessor, self).__init__(dataset_path, is_interactive, obs_frame_num,
                                              pred_frame_num, skip_frame_num, use_video_once, custom_name)
        self.dataset_total_frame_num = 900
        self.is_3d = is_3d
        if is_3d:
            self.start_dim = 5
            self.end_dim = 8
        else:
            self.start_dim = 3
            self.end_dim = 5
        if self.is_3d:
            if self.is_interactive:
                self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, 'JTA_interactive', '3D')
            else:
                self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, 'JTA', '3D')
        else:
            if self.is_interactive:
                self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, 'JTA_interactive', '2D')
            else:
                self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, 'JTA', '2D')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.meta_data = {
            'avg_person': [],
            'max_pose': np.zeros(3) if self.is_3d else np.zeros(2),
            'min_pose': np.array([1000.0, 1000.0, 1000.0]) if self.is_3d else np.zeros(2),
            'count': 0,
            'sum2_pose': np.zeros(3) if self.is_3d else np.zeros(2),
            'sum_pose': np.zeros(3) if self.is_3d else np.zeros(2)
        }

    def __generate_image_path(self, frame_num, file_name, input_matrix, total_frame_num):
        image_relative_path = re.search("(seq_\d+).json", file_name).group(1)
        video_data = {
            'obs_frames': defaultdict(list),
            'future_frames': defaultdict(list)
        }
        obs_frames = []
        future_frames = []
        for j in range(1, total_frame_num * (self.skip_frame_num + 1) + 1, self.skip_frame_num + 1):
            frame_data = defaultdict(list)
            frame = input_matrix[
                input_matrix[:, 0] == frame_num * total_frame_num * (self.skip_frame_num + 1) + j]  # find frame data
            for pose in frame:
                frame_data[pose[1]] = pose[0]
            for p_id in frame_data.keys():
                if j <= self.obs_frame_num * (self.skip_frame_num + 1):
                    video_data['obs_frames'][p_id].append(frame_data[p_id])
                else:
                    video_data['future_frames'][p_id].append(frame_data[p_id])
        for p_id in video_data['obs_frames']:
            if p_id in video_data['future_frames'].keys() and video_data['obs_frames'][
                p_id].__len__() == self.obs_frame_num and \
                    video_data['future_frames'][
                        p_id].__len__() == self.pred_frame_num:
                obs_frames.append(video_data['obs_frames'][p_id])
                future_frames.append(video_data['future_frames'][p_id])
        for p_id in range(len(obs_frames)):
            for j in range(len(obs_frames[0])):
                obs_frames[p_id][j] = f'{image_relative_path}/{int(obs_frames[p_id][j])}.jpg'
        for p_id in range(len(future_frames)):
            for j in range(len(future_frames[0])):
                future_frames[p_id][j] = f'{image_relative_path}/{int(future_frames[p_id][j])}.jpg'
        return obs_frames, future_frames

    def normal(self, data_type='train'):
        count_total_data = 0
        count_mask_data = 0
        logger.info('start creating JTA normal static data ... ')
        if self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.jsonl'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_JTA.jsonl'
        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"
        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.dataset_total_frame_num // (
                total_frame_num * (self.skip_frame_num + 1)) if not self.use_video_once else 1

        for entry in os.scandir(self.dataset_path):
            if not entry.path.endswith('.json'):
                continue
            with open(entry.path, 'r') as json_file:
                logger.info(f'file name: {entry.name}')
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
                    for j in range(1, total_frame_num * (self.skip_frame_num + 1) + 1, self.skip_frame_num + 1):
                        frame_data = {
                            'pose': defaultdict(list),
                            'mask': defaultdict(list)
                        }
                        frame = matrix[
                            matrix[:, 0] == i * total_frame_num * (self.skip_frame_num + 1) + j]  # find frame data
                        for pose in frame:
                            masked = 0
                            # pose data
                            for kp_position in range(self.start_dim, self.end_dim):
                                frame_data['pose'][pose[1]].append(pose[kp_position])
                            # mask data
                            for masking_state in range(9, 10):
                                masked += pose[masking_state]
                            frame_data['mask'][pose[1]].append(1 if masked > 0 else 0)
                            count_mask_data += 1 if masked else  0
                            count_total_data += 1
                        for p_id in frame_data['pose'].keys():
                            if j <= self.obs_frame_num * (self.skip_frame_num + 1):
                                video_data['obs_pose'][p_id].append(frame_data['pose'][p_id])
                                video_data['obs_mask'][p_id].append(frame_data['mask'][p_id])
                            else:
                                video_data['future_pose'][p_id].append(frame_data['pose'][p_id])
                                video_data['future_mask'][p_id].append(frame_data['mask'][p_id])
                    for p_id in video_data['obs_pose']:
                        if p_id in video_data['future_pose'].keys() \
                                and video_data['obs_pose'][p_id].__len__() == self.obs_frame_num \
                                and video_data['future_pose'][p_id].__len__() == self.pred_frame_num:
                            obs.append(video_data['obs_pose'][p_id])
                            obs_mask.append(video_data['obs_mask'][p_id])
                            future.append(video_data['future_pose'][p_id])
                            future_mask.append(video_data['future_mask'][p_id])
                    obs_frames, future_frames = self.__generate_image_path(i, entry.name, matrix, total_frame_num)
                    if len(obs) > 0:
                        # max_acceptable_len = max(len(obs), len(future), len(obs_mask), len(obs_frames))
                        if data_type == 'train':
                            self.update_meta_data(self.meta_data, obs, 3 if self.is_3d else 2)
                        if not self.is_interactive:
                            for p_id in range(len(obs)):
                                data.append([
                                    '%s-%d' % (video_number, i), obs[p_id], future[p_id], obs_mask[p_id],
                                    future_mask[p_id], obs_frames[p_id], future_frames[p_id]
                                ])
                        else:
                            data.append([
                                '%s-%d' % (video_number, i), obs, future, obs_mask, future_mask,
                                obs_frames[0], future_frames[0]
                            ])
                with jsonlines.open(os.path.join(self.output_dir, output_file_name), 'a') as writer:
                    for data_row in data:
                        writer.write({
                            'video_section': data_row[0],
                            'observed_pose': data_row[1],
                            'future_pose': data_row[2],
                            'observed_mask': data_row[3],
                            'future_mask': data_row[4],
                            'observed_image_path': data_row[5],
                            'future_image_path': data_row[6]
                        })
        print("mask percentage: {}".format(count_mask_data / count_total_data))
        self.meta_data['mask_percentage'] = count_mask_data / count_total_data

        pose_format = '3D' if self.is_3d else '2D'
        self.save_meta_data(self.meta_data, self.output_dir, pose_format, data_type)
        #self.save_meta_data(self.meta_data, self.output_dir, self.is_3d, data_type)
