import logging
import os

import jsonlines
import numpy as np
from utils.others import AMASSconvertTo3D

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor

logger = logging.getLogger(__name__)

amass_splits = {
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'TCD_handMocap', 'ACCAD'],
    'validation': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['BioMotionLab_NTroje'],
}

class AmassPreprocessor(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name, save_total_frames):
        super(AmassPreprocessor, self).__init__(dataset_path, is_interactive, obs_frame_num,
                                               pred_frame_num, skip_frame_num, use_video_once,
                                               custom_name, save_total_frames)

        assert not is_interactive, 'this dataset does not support interactive'

        self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, 'AMASS')
        if self.is_interactive:
            self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, 'AMASS_interactive')
        elif self.save_total_frames:
            self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, 'AMASS_total')

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

    def normal(self, data_type='train'):
        logger.info('start creating AMASS normal static data ... ')
        total_frame_num = self.obs_frame_num + self.pred_frame_num

        if self.save_total_frames:
            if self.custom_name:
                output_file_name = f'{data_type}_xyz_{self.custom_name}.jsonl'
            else:
                output_file_name = f'{data_type}_xyz_AMASS.jsonl'
        elif self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.jsonl'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_AMASS.jsonl'
        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"

        assert data_type in amass_splits, "data type must be one of train, validation or test"
        
        dataset_names = amass_splits[data_type]
        for dataset_name in dataset_names:
            raw_dataset_name = dataset_name
            logger.info(f'dataset name: {dataset_name}')
            for sub in os.listdir(os.path.join(self.dataset_path, dataset_name)):
                raw_sub = sub
                logger.info(f'subject name: {sub}')
                sub = os.path.join(self.dataset_path, dataset_name, sub)
                if not os.path.isdir(sub):
                    continue
                for act in os.listdir(sub):
                    if not act.endswith('.npz'):
                        continue
                    raw_act = act[:-4]
                    pose_all = np.load(os.path.join(sub, act))
                    try:
                        pose_data = pose_all['poses']
                    except:
                        print('no poses at {} {}'.format(sub, act))
                        continue

                    pose_data = AMASSconvertTo3D(pose_data) # shape = [num frames , 66]
                    section_range = pose_data.shape[0] // (
                            total_frame_num * (self.skip_frame_num + 1)) if self.use_video_once is False else 1 

                    if self.save_total_frames:
                        section_range = 1
                        total_frame_num = pose_data.shape[0]
                        self.obs_frame_num = total_frame_num
                        self.pred_frame_num = 0
                        self.skip_frame_num = 0

                    data = []
                    for i in range(section_range):
                        video_data = {
                            'obs_pose': list(),
                            'future_pose': list(),
                            'fps': int(pose_all['mocap_framerate'].item())
                        }

                        for j in range(0, total_frame_num * (self.skip_frame_num + 1), self.skip_frame_num + 1):
                            if j <= (self.skip_frame_num + 1) * self.obs_frame_num:
                                video_data['obs_pose'].append(
                                    (pose_data[i * total_frame_num * (self.skip_frame_num + 1) + j,12:]*1000).tolist()
                                )
                            else:
                                video_data['future_pose'].append(
                                    (pose_data[i * total_frame_num * (self.skip_frame_num + 1) + j,12:]*1000).tolist()
                                )
                        
                        if data_type == 'train':
                            self.update_meta_data(self.meta_data, list(video_data['obs_pose']), 3)
                        data.append([
                            '%s-%d' % ("{}-{}-{}".format(raw_dataset_name, raw_sub, raw_act), i),
                            video_data['obs_pose'], video_data['future_pose'], video_data['fps']
                        ])
                          
                    with jsonlines.open(os.path.join(self.output_dir, output_file_name), 'a') as writer:
                        for data_row in data:
                            if not self.save_total_frames:
                                writer.write({
                                    'video_section': data_row[0],
                                    'observed_pose': data_row[1],
                                    'future_pose': data_row[2],
                                    'fps': data_row[3]
                                })
                            else:
                                writer.write({
                                    'video_section': data_row[0],
                                    'xyz_pose': data_row[1],
                                    'fps': data_row[3]
                                })

        self.save_meta_data(self.meta_data, self.output_dir, True, data_type)