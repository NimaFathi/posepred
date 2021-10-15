import logging
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor
from utils.others import DATA_FORMAT

logger = logging.getLogger(__name__)


class Preprocessor3DPW(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name):
        super(Preprocessor3DPW, self).__init__(dataset_path, is_interactive, obs_frame_num,
                                               pred_frame_num, skip_frame_num, use_video_once, custom_name)

        self.output_dir = os.path.join(
            PREPROCESSED_DATA_DIR, '3DPW_interactive') if self.is_interactive else os.path.join(
            PREPROCESSED_DATA_DIR, '3DPW'
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.meta_data = {
            'avg_person': [],
            'count': 0,
            'sum2_pose': np.zeros(3),
            'sum_pose': np.zeros(3)
        }
        self.hdf_keys_dict = {
            0: 'video_section', 1: 'observed_pose', 2: 'future_pose', 3: 'observed_image_path',
            4: 'future_image_path', 5: 'observed_cam_extrinsic', 6: 'future_cam_extrinsic', 7: 'cam_intrinsic'
        }

    def normal(self, data_type='train'):
        logger.info('start creating 3DPW normal static data ... ')
        total_frame_num = self.obs_frame_num + self.pred_frame_num

        if self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.{DATA_FORMAT}'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_3dpw.{DATA_FORMAT}'
        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"
        hf, hf_groups = self.init_hdf(output_file_name)
        for entry in os.scandir(self.dataset_path):
            if not entry.name.endswith('.pkl'):
                continue
            logger.info(f'file name: {entry.name}')
            pickle_obj = pd.read_pickle(entry.path)
            video_name = re.search('(\w+).pkl', entry.name).group(1)
            pose_data = np.array(pickle_obj['jointPositions'])
            frame_ids_data = pickle_obj['img_frame_ids']
            cam_extrinsic = pickle_obj['cam_poses'][:, :3]
            cam_intrinsic = pickle_obj['cam_intrinsics'].tolist()
            section_range = pose_data.shape[1] // (
                    total_frame_num * (self.skip_frame_num + 1)) if self.use_video_once is False else 1
            data = []
            for i in range(section_range):
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
                            ])
                    else:
                        data.append([
                            '%s-%d' % (video_name, i),
                            list(video_data['obs_pose'].values()), list(video_data['future_pose'].values()),
                            video_data['obs_frames'][0], video_data['future_frames'][0],
                            video_data['obs_cam_ext'], video_data['future_cam_ext'], cam_intrinsic
                        ])
            self.update_hdf(hf_groups, data)
        self.save_meta_data(self.meta_data, self.output_dir, True, data_type)
        hf.close()
