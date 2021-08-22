import csv
import json
import os
import numpy as np
from collections import defaultdict

from preprocessor.preprocessor import Processor, OUTPUT_DIR


class SoMoFPoseTrackPreprocessor(Processor):
    def __init__(self, mask, dataset_path, is_disentangle, is_interactive, obs_frame_num, pred_frame_num,
                 skip_frame_num,
                 use_video_once, custom_name):
        super(SoMoFPoseTrackPreprocessor, self).__init__(dataset_path, is_disentangle, is_interactive, obs_frame_num,
                                                         pred_frame_num, skip_frame_num, use_video_once, custom_name)
        self.mask = mask
        self.output_dir = os.path.join(OUTPUT_DIR, 'SoMoF_PoseTrack_interactive') if self.is_interactive else os.path.join(
            OUTPUT_DIR, 'SoMoF_PoseTrack')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def normal(self, data_type='train'):
        preprocessed_data = self.__clean_data(data_type)
        self.__save_csv(data_type, preprocessed_data)

    def disentangle(self, data_type='train'):
        preprocessed_data = self.__clean_data(data_type)
        self.__create_disentangle_data(preprocessed_data['obs_pose'])
        local_data = defaultdict(np.array)
        global_data = defaultdict(np.array)
        if data_type != 'test':
            self.__create_disentangle_data(preprocessed_data['future_pose'])
        local_obs_pose = []
        local_obs_mask = []
        global_obs_mask = []
        global_obs_pose = []
        for i in range(len(preprocessed_data['obs_pose'])):
            local_obs_pose.append(np.array(preprocessed_data['obs_pose'][i])[:, :, 2:].tolist())
            local_obs_mask.append(np.array(preprocessed_data['obs_mask'][i])[:, :, 1:].tolist())
            global_obs_pose.append(np.array(preprocessed_data['obs_pose'][i])[:, :, :2])
            global_obs_mask.append(np.array(preprocessed_data['obs_mask'][i])[:, :, :1])
        local_data['obs_pose'] = np.array(local_obs_pose, dtype=object)
        global_data['obs_pose'] = np.array(global_obs_pose, dtype=object)
        local_data['obs_mask'] = np.array(local_obs_mask, dtype=object)
        global_data['obs_mask'] = np.array(global_obs_mask, dtype=object)
        local_data['obs_frames_path'] = preprocessed_data['obs_frames_path']
        global_data['obs_frames_path'] = preprocessed_data['obs_frames_path']
        if data_type != 'test':
            local_future_pose = []
            global_future_pose = []
            local_future_mask = []
            global_future_mask = []
            for i in range(len(preprocessed_data['obs_pose'])):
                local_future_pose.append(np.array(preprocessed_data['future_pose'][i])[:, :, 2:].tolist())
                local_future_mask.append(np.array(preprocessed_data['future_mask'][i])[:, :, 1:])
                global_future_pose.append(np.array(preprocessed_data['future_pose'][i])[:, :, :2].tolist())
                global_future_mask.append(np.array(preprocessed_data['future_mask'][i])[:, :, :1])
            local_data['future_pose'] = np.array(local_future_pose, dtype=object)
            global_data['future_pose'] = np.array(global_future_pose, dtype=object)
            local_data['future_mask'] = np.array(local_future_mask, dtype=object)
            global_data['future_mask'] = np.array(global_future_mask, dtype=object)
        self.__save_csv(data_type,  global_data, 'global')
        self.__save_csv(data_type,  local_data, 'local')

    def __save_csv(self, data_type, processed_data, file_type=None):
        if self.custom_name:
            if file_type is None:
                output_name = f'{data_type}_16_14_1_{self.custom_name}.csv'
            else:
                output_name = f'{file_type}_{data_type}_16_14_1_{self.custom_name}.csv'
        else:
            if file_type is None:
                output_name = f'{data_type}_16_14_1_SoMoF_PoseTrack.csv'
            else:
                output_name = f'{file_type}_{data_type}_16_14_1_SoMoF_PoseTrack.csv'
        data = []
        if data_type == 'test':
            header = ['video_section', 'observed_pose', 'observed_mask', 'obs_frames']
        else:
            header = ['video_section', 'observed_pose', 'future_pose', 'observed_mask', 'future_mask', 'obs_frames']
        with open(os.path.join(self.output_dir, output_name), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        if data_type == 'test':
            if self.is_interactive:
                for vid_id in range(len(processed_data['obs_pose'])):
                    data.append(['%d-%d' % (vid_id, 0),
                                 processed_data['obs_pose'][vid_id],
                                 processed_data['obs_mask'][vid_id],
                                 processed_data['obs_frames_path'][vid_id]
                                 ])
            else:
                for vid_id in range(len(processed_data['obs_pose'])):
                    for p_id in range(len(processed_data['future_pose'][vid_id])):
                        data.append([
                            '%d-%d' % (vid_id, 0), processed_data['obs_pose'][vid_id][p_id],
                            processed_data['obs_mask'][vid_id][p_id],
                            processed_data['obs_frames_path'][vid_id]
                        ])
        else:
            if self.is_interactive:
                for vid_id in range(processed_data['obs_pose'].__len__()):
                    data.append([
                        '%d-%d' % (vid_id, 0), processed_data['obs_pose'][vid_id],
                        processed_data['future_pose'][vid_id],
                        processed_data['obs_mask'][vid_id],
                        processed_data['future_mask'][vid_id], processed_data['obs_frames_path'][vid_id]
                    ])
            else:
                for vid_id in range(len(processed_data['obs_pose'])):
                    for p_id in range(len(processed_data['future_pose'][vid_id])):
                        data.append([
                            '%d-%d' % (vid_id, 0),
                            processed_data['obs_pose'][vid_id][p_id], processed_data['future_pose'][vid_id][p_id],
                            processed_data['obs_mask'][vid_id][p_id], processed_data['future_pose'][vid_id][p_id],
                            processed_data['obs_frames_path'][vid_id]
                        ])
        with open(os.path.join(self.is_interactive, output_name), 'a') as f_object:
            writer = csv.writer(f_object)
            writer.writerows(data)

    def __clean_data(self, data_type):
        if data_type == 'validation':
            data_type = 'valid'
        files_names = defaultdict()
        processed_data = defaultdict(np.array)
        files_names['obs_pose'] = f'posetrack_{data_type}_in.json'
        files_names['obs_mask'] = f'posetrack_{data_type}_masks_in.json'
        files_names['obs_frames_path'] = f'posetrack_{data_type}_frames_in.json'
        if data_type == 'train' or data_type == 'valid':
            files_names['future_pose'] = f'posetrack_{data_type}_out.json'
            files_names['future_mask'] = f'posetrack_{data_type}_masks_out.json'
        for file_name_key, file_name in files_names.items():
            with open(os.path.join(self.dataset_path, file_name), 'r') as json_file:
                processed_data[file_name_key] = np.array(json.load(json_file), dtype=object)
        return processed_data

    @staticmethod
    def __create_disentangle_data(data):
        for vid_id in range(len(data)):
            vid_data = np.array(data[vid_id])
            for p_id in range(vid_data.shape[0]):
                neck_joint_data = vid_data[p_id][:, :2]
                other_joints_data = vid_data[p_id][:, 2:]
                for i, val in enumerate(other_joints_data):
                    for j in range(13):
                        if not np.all((val[2 * j: 2 * (j + 1)] == 0)):
                            val[2 * j: 2 * (j + 1)] = np.subtract(val[2 * j: 2 * (j + 1)], neck_joint_data[i])
            data[vid_id] = vid_data
        return data
