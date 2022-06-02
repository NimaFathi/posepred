import logging
import os
import re
from collections import defaultdict

import jsonlines
import numpy as np
import pandas as pd
from utils.others import DPWconvertTo3D

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor

logger = logging.getLogger(__name__)


class Preprocessor3DPW(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name, save_total_frames, load_60Hz=False):
        super(Preprocessor3DPW, self).__init__(dataset_path, is_interactive, obs_frame_num,
                                               pred_frame_num, skip_frame_num, use_video_once,
                                               custom_name, save_total_frames)

        self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, '3DPW')
        if self.is_interactive:
            self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, '3DPW_interactive')
        elif self.save_total_frames:
            self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, '3DPW_total')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.meta_data = {
            'avg_person': [],
            'max_pose': np.zeros(3),
            'min_pose': np.array([1000.0, 1000.0, 1000.0]),
            'count': 0,
            'sum2_pose': np.zeros(3),
            'sum_pose': np.zeros(3)
        }

        self.load_60Hz = load_60Hz

    def normal(self, data_type='train'):
        logger.info('start creating 3DPW normal static data ... ')
        total_frame_num = self.obs_frame_num + self.pred_frame_num

        if self.save_total_frames:
            if self.custom_name:
                output_file_name = f'{data_type}_xyz_{self.custom_name}.jsonl'
            else:
                output_file_name = f'{data_type}_xyz_3dpw.jsonl'
        elif self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.jsonl'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_3dpw.jsonl'
        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"

  
        self.dataset_path = os.path.join(self.dataset_path, data_type)

        for entry in os.scandir(self.dataset_path):
            if not entry.name.endswith('.pkl'):
                continue
            logger.info(f'file name: {entry.name}')
            pickle_obj = pd.read_pickle(entry.path)
            video_name = re.search('(\w+).pkl', entry.name).group(1)
            if self.load_60Hz:
                pose_data = np.array(pickle_obj['poses_60Hz'])
            else:
                pose_data = np.array(pickle_obj['jointPositions'])
                frame_ids_data = pickle_obj['img_frame_ids']
                cam_extrinsic = pickle_obj['cam_poses'][:, :3]
                cam_intrinsic = pickle_obj['cam_intrinsics'].tolist()

            pose_data = DPWconvertTo3D(pose_data)
            section_range = pose_data.shape[1] // (
                    total_frame_num * (self.skip_frame_num + 1)) if self.use_video_once is False else 1 

            if self.save_total_frames:
                section_range = 1
                total_frame_num = pose_data.shape[1]
                self.obs_frame_num = total_frame_num
                self.pred_frame_num = 0
                self.skip_frame_num = 0

            data = []
            for i in range(section_range):
                video_data = {
                    'obs_pose': defaultdict(list),
                    'future_pose': defaultdict(list),
                }
                if not self.load_60Hz:
                    video_data = {
                        'obs_pose': defaultdict(list),
                        'future_pose': defaultdict(list),
                        'obs_frames': defaultdict(list),
                        'future_frames': defaultdict(list),
                        'obs_cam_ext': list(),
                        'future_cam_ext': list()
                    }
                for j in range(1, total_frame_num * (self.skip_frame_num + 1) + 1, self.skip_frame_num + 1):
                    for p_id in range(pose_data.shape[0]):
                        if j <= (self.skip_frame_num + 1) * self.obs_frame_num:
                            video_data['obs_pose'][p_id].append(
                                pose_data[p_id, i * total_frame_num * (self.skip_frame_num + 1) + j - 1, :].tolist()
                            )
                            if not self.load_60Hz:
                                video_data['obs_frames'][p_id].append(
                                    f'{video_name}/image_{i * total_frame_num * (self.skip_frame_num + 1) + j - 1:05}.jpg'
                                )
                                if p_id == 0:
                                    video_data['obs_cam_ext'].append(
                                        cam_extrinsic[i * total_frame_num * (self.skip_frame_num + 1) + j - 1].tolist()
                                    )
                        else:
                            video_data['future_pose'][p_id].append(
                                pose_data[p_id, i * total_frame_num * (self.skip_frame_num + 1) + j - 1, :].tolist()
                            )
                            if not self.load_60Hz:
                                video_data['future_frames'][p_id].append(
                                    f'{video_name}/image_{i * total_frame_num * (self.skip_frame_num + 1) + j - 1:05}.jpg'
                                )
                                if p_id == 0:
                                    video_data['future_cam_ext'].append(
                                        cam_extrinsic[i * total_frame_num * (self.skip_frame_num + 1) + j - 1].tolist()
                                    )
                if len(list(video_data['obs_pose'].values())) > 0:
                    if data_type == 'train':
                        self.update_meta_data(self.meta_data, list(video_data['obs_pose'].values()), 3)
                    if not self.is_interactive:
                        for p_id in range(len(pose_data)):
                            data.append([
                                '%s-%d' % (video_name, i),
                                video_data['obs_pose'][p_id], video_data['future_pose'][p_id],
                                video_data['obs_frames'][p_id], video_data['future_frames'][p_id],
                                video_data['obs_cam_ext'], video_data['future_cam_ext'], cam_intrinsic
                            ] if not self.load_60Hz else [
                                '%s-%d' % (video_name, i),
                                video_data['obs_pose'][p_id], video_data['future_pose'][p_id]
                            ])
                    else:
                        data.append([
                            '%s-%d' % (video_name, i),
                            list(video_data['obs_pose'].values()), list(video_data['future_pose'].values()),
                            video_data['obs_frames'][0], video_data['future_frames'][0],
                            video_data['obs_cam_ext'], video_data['future_cam_ext'], cam_intrinsic
                        ] if not self.load_60Hz else [
                                 '%s-%d' % (video_name, i),
                            list(video_data['obs_pose'].values()), list(video_data['future_pose'].values()),
                            ])
            with jsonlines.open(os.path.join(self.output_dir, output_file_name), 'a') as writer:
                for data_row in data:
                    if not self.save_total_frames:
                        if not self.load_60Hz:
                            writer.write({
                                'video_section': data_row[0],
                                'observed_pose': data_row[1],
                                'future_pose': data_row[2],
                                'observed_image_path': data_row[3],
                                'future_image_path': data_row[4],
                                'observed_cam_extrinsic': data_row[5],
                                'future_cam_extrinsic': data_row[6],
                                'cam_intrinsic': data_row[7]
                            })
                        else:
                             writer.write({
                                'video_section': data_row[0],
                                'observed_pose': data_row[1],
                                'future_pose': data_row[2]
                            })
                    else:
                        if not self.load_60Hz:
                            writer.write({
                                'video_section': data_row[0],
                                'total_pose': data_row[1],
                                'total_image_path': data_row[3],
                                'total_cam_extrinsic': data_row[5],
                                'cam_intrinsic': data_row[7]
                            })
                        else:
                             writer.write({
                                'video_section': data_row[0],
                                'total_pose': data_row[1],
                            })

        self.save_meta_data(self.meta_data, self.output_dir, True, data_type)

