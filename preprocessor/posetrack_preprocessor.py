import json
import logging
import os
from collections import defaultdict

import jsonlines
import numpy as np

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor
from utils.others import DATA_FORMAT

logger = logging.getLogger(__name__)


class PoseTrackPreprocessor(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num,
                 skip_frame_num, use_video_once, custom_name):
        super(PoseTrackPreprocessor, self).__init__(dataset_path, is_interactive, obs_frame_num,
                                                    pred_frame_num, skip_frame_num, use_video_once, custom_name)
        self.dataset_total_frame_num = 30
        self.output_dir = os.path.join(
            PREPROCESSED_DATA_DIR, 'PoseTrack_interactive') if self.is_interactive else os.path.join(
            PREPROCESSED_DATA_DIR, 'PoseTrack'
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.meta_data = {
            'avg_person': [],
            'count': 0,
            'sum2_pose': np.zeros(2),
            'sum_pose': np.zeros(2)
        }
        self.hdf_keys_dict = {0: 'video_section', 1: 'observed_pose', 2: 'future_pose', 3: 'observed_mask',
                              4: 'future_mask',
                              5: 'observed_image_path', 6: 'future_image_path'}

    def __generate_image_path(self, json_data, frame_ids, total_frame_num):
        video_dict = {
            'obs_frames': defaultdict(list),
            'future_frames': defaultdict(list)
        }
        obs_frames = []
        future_frames = []
        for j in range(1, total_frame_num * (self.skip_frame_num + 1) + 1, self.skip_frame_num + 1):
            for pedestrian in frame_ids.keys():
                if frame_ids[pedestrian].__len__() < j:
                    continue
                if j <= self.obs_frame_num * (self.skip_frame_num + 1):
                    video_dict['obs_frames'][pedestrian].append(frame_ids[pedestrian][j - 1])
                else:
                    video_dict['future_frames'][pedestrian].append(frame_ids[pedestrian][j - 1])
        for p_id in video_dict['obs_frames'].keys():
            if p_id in video_dict['future_frames'].keys() \
                    and video_dict['obs_frames'][p_id].__len__() == self.obs_frame_num \
                    and video_dict['future_frames'][p_id].__len__() == self.pred_frame_num:
                obs_frames.append(video_dict['obs_frames'][p_id])
                future_frames.append(video_dict['future_frames'][p_id])
        images = json_data.get('images')
        for image in images:
            image_id = image.get('frame_id')
            for indx in range(len(obs_frames)):
                for j, frame_id in enumerate(obs_frames[indx]):
                    if frame_id == image_id:
                        obs_frames[indx][j] = image.get('file_name')
            for indx in range(len(future_frames)):
                for j, frame_id in enumerate(future_frames[indx]):
                    if frame_id == image_id:
                        future_frames[indx][j] = image.get('file_name')
        return obs_frames, future_frames

    def normal(self, data_type='train'):
        logger.info('start creating PoseTrack normal static data ... ')
        if self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.{DATA_FORMAT}'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_PoseTrack{DATA_FORMAT}'
        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"
        hf, hf_groups = self.init_hdf(hdf_keys=hdf_keys, file_name=output_file_name)
        total_frame_num = self.obs_frame_num + self.pred_frame_num
        for entry in os.scandir(self.dataset_path):
            if not entry.path.endswith('.json'):
                continue
            with open(entry.path, 'r') as json_file:
                json_data = json.load(json_file)
                annotations = json_data.get('annotations')
                section_range = json_data['images'][0]['nframes'] // (
                        total_frame_num * 1) if self.use_video_once is False else 1
                if not annotations:
                    continue
                logger.info(f'file name: {entry.name}')
                video_id = json_data.get('images')[0].get('vid_id')
                pose = defaultdict(list)
                mask = defaultdict(list)
                frame_ids = defaultdict(list)
                data = []
                for annotation in annotations:
                    frame_pose = []
                    frame_mask = []
                    keypoints = annotation.get('keypoints')
                    image_id = annotation.get('image_id')
                    for i in range(1, len(keypoints) + 1):
                        if i % 3 == 0:
                            frame_mask.append(keypoints[i - 1])
                        else:
                            frame_pose.append(keypoints[i - 1])
                    pose[annotation.get('track_id')].append(frame_pose)
                    frame_ids[annotation.get('track_id')].append(image_id)
                    mask[annotation.get('track_id')].append(frame_mask)
                for i in range(section_range):
                    video_dict = {
                        'obs_pose': defaultdict(list),
                        'future_pose': defaultdict(list),
                        'obs_mask': defaultdict(list),
                        'future_mask': defaultdict(list),
                    }
                    obs_pose = []
                    future_pose = []
                    obs_mask = []
                    future_mask = []
                    for j in range(1, total_frame_num * (self.skip_frame_num + 1) + 1, self.skip_frame_num + 1):
                        for pedestrian in pose.keys():
                            if pose[pedestrian].__len__() < j:
                                continue
                            if j <= self.obs_frame_num * (self.skip_frame_num + 1):
                                video_dict['obs_pose'][pedestrian].append(pose[pedestrian][j - 1])
                                video_dict['obs_mask'][pedestrian].append(mask[pedestrian][j - 1])
                            else:
                                video_dict['future_pose'][pedestrian].append(pose[pedestrian][j - 1])
                                video_dict['future_mask'][pedestrian].append(mask[pedestrian][j - 1])
                    for p_id in video_dict['obs_pose'].keys():
                        if p_id in video_dict['future_pose'].keys() \
                                and video_dict['obs_pose'][p_id].__len__() == self.obs_frame_num \
                                and video_dict['future_pose'][p_id].__len__() == self.pred_frame_num:
                            obs_pose.append(video_dict['obs_pose'][p_id])
                            obs_mask.append(video_dict['obs_mask'][p_id])
                            future_pose.append(video_dict['future_pose'][p_id])
                            future_mask.append(video_dict['future_mask'][p_id])
                    obs_frames, future_frames = self.__generate_image_path(json_data, frame_ids, total_frame_num)
                    if len(obs_pose) > 0:
                        if data_type == 'train':
                            self.update_meta_data(self.meta_data, obs_pose, 2)
                        if not self.is_interactive:
                            for p_id in range(len(obs_pose)):
                                data.append(
                                    [
                                        '%s-%d' % (video_id, i), obs_pose[p_id], future_pose[p_id], obs_mask[p_id],
                                        future_mask[p_id], obs_frames[p_id], future_frames[p_id]
                                    ]
                                )
                        else:
                            data.append(
                                [
                                    '%s-%d' % (video_id, i), obs_pose, future_pose, obs_mask, future_mask,
                                    obs_frames[0], future_frames[0]
                                ]
                            )
                with jsonlines.open(os.path.join(self.output_dir, output_file_name), 'a') as writer:
                    for data_row in data:
                        writer.write({
                            'video_section': data_row[0],
                            'observed_pose': data_row[1],
                            'future_pose': data_row[2],
                            'observed_mask': data_row[3],
                            'future_mask': data_row[4],
                            'observed_image_path': data_row[5],
                            'future_image_path': data_row[6],
                        })
                self.update_hdf(hf_groups, data)

        hf.close()
        self.save_meta_data(self.meta_data, self.output_dir, False, data_type)
