import csv
import json
import os
from collections import defaultdict

from preprocessor.preprocessor import Processor, OUTPUT_DIR


class PoseTrackPreprocessor(Processor):
    def __init__(self, mask, dataset_path, is_disentangle, is_interactive, obs_frame_num, pred_frame_num,
                 skip_frame_num,
                 use_video_once, custom_name):
        super(PoseTrackPreprocessor, self).__init__(dataset_path, is_disentangle, is_interactive, obs_frame_num,
                                                    pred_frame_num, skip_frame_num, use_video_once, custom_name)
        self.dataset_total_frame_num = 30
        self.mask = mask
        self.output_dir = os.path.join(OUTPUT_DIR, 'PoseTrack_interactive') if self.is_interactive else os.path.join(
            OUTPUT_DIR, 'PoseTrack')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def __generate_image_path(self, json_data, frame_ids, total_frame_num):
        video_dict = {
            'obs_frames': defaultdict(list),
            'future_frames': defaultdict(list)
        }
        obs_frames = []
        future_frames = []
        for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
            for pedestrian in frame_ids.keys():
                if frame_ids[pedestrian].__len__() < j:
                    continue
                if j <= self.obs_frame_num * self.skip_frame_num:
                    video_dict['obs_frames'][pedestrian].append(frame_ids[pedestrian][j - 1])
                else:
                    video_dict['future_frames'][pedestrian].append(frame_ids[pedestrian][j - 1])
        for p_id in video_dict['obs_frames'].keys():
            if p_id in video_dict['future_frames'].keys() and video_dict['obs_frames'][
                p_id].__len__() == self.obs_frame_num and \
                    video_dict['future_frames'][p_id].__len__() == self.pred_frame_num:
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
        print('start creating PoseTrack normal static data ... ')
        header = ['video_section', 'observed_pose', 'future_pose', 'observed_mask', 'future_mask',
                  'obs_frames_related_path', 'future_frames_related_path']
        if self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.csv'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_PoseTrack.csv'
        with open(os.path.join(self.output_dir, output_file_name), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = 30 // (total_frame_num * 1) if self.use_video_once is False else 1

        for entry in os.scandir(self.dataset_path):
            if not entry.path.endswith('.json'):
                continue
            with open(entry.path, 'r') as json_file:
                json_data = json.load(json_file)
                annotations = json_data.get('annotations')
                if not annotations:
                    continue
                print(f'file name: {entry.name}')
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
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        for pedestrian in pose.keys():
                            if pose[pedestrian].__len__() < j:
                                continue
                            if j <= self.obs_frame_num * self.skip_frame_num:
                                video_dict['obs_pose'][pedestrian].append(pose[pedestrian][j - 1])
                                video_dict['obs_mask'][pedestrian].append(mask[pedestrian][j - 1])
                            else:
                                video_dict['future_pose'][pedestrian].append(pose[pedestrian][j - 1])
                                video_dict['future_mask'][pedestrian].append(mask[pedestrian][j - 1])
                    for p_id in video_dict['obs_pose'].keys():
                        if p_id in video_dict['future_pose'].keys() and video_dict['obs_pose'][
                            p_id].__len__() == self.obs_frame_num and \
                                video_dict['future_pose'][p_id].__len__() == self.pred_frame_num:
                            obs_pose.append(video_dict['obs_pose'][p_id])
                            obs_mask.append(video_dict['obs_mask'][p_id])
                            future_pose.append(video_dict['future_pose'][p_id])
                            future_mask.append(video_dict['future_mask'][p_id])
                    obs_frames, future_frames = self.__generate_image_path(json_data, frame_ids, total_frame_num)
                    if not self.is_interactive:
                        for p_id in range(len(obs_pose)):
                            data.append(['%s-%d' % (video_id, i), obs_pose[p_id], future_pose[p_id], obs_mask[p_id],
                                         future_mask[p_id], obs_frames[p_id], future_frames[p_id]])
                    else:
                        data.append(['%s-%d' % (video_id, i), obs_pose, future_pose, obs_mask, future_mask, obs_frames,
                                     future_frames])
                with open(os.path.join(self.output_dir, output_file_name), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data)

    def disentangle(self, data_type='train'):
        print('start creating PoseTrack disentangle static data ... ')
        self.disentangle_global(data_type)
        self.disentangle_local(data_type)

    def disentangle_global(self, data_type='train'):
        header = ['video_section', 'observed_pose', 'future_pose', 'observed_mask', 'future_mask',
                  'obs_frames_related_path', 'future_frames_related_path']
        if self.custom_name:
            global_file_name = f'global_{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.csv'
        else:
            global_file_name = f'global_{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_PoseTrack.csv'
        with open(os.path.join(self.output_dir, global_file_name), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.dataset_total_frame_num // (
                total_frame_num * self.skip_frame_num) if self.use_video_once is False else 1

        for entry in os.scandir(self.dataset_path):
            if not entry.path.endswith('.json'):
                continue
            with open(entry.path, 'r') as json_file:
                json_data = json.load(json_file)
                annotations = json_data.get('annotations')
                if not annotations:
                    continue
                print('global: {}'.format(entry.name))
                video_id = json_data.get('images')[0].get('vid_id')
                frame_global_data = {
                    'pose': defaultdict(list),
                    'mask': defaultdict(list),
                    'frame_ids': defaultdict(list)
                }
                data = []
                for annotation in annotations:
                    keypoints = annotation.get('keypoints')
                    frame_global_data['frame_ids'][annotation.get('track_id')].append(annotation.get('image_id'))
                    frame_global_data['pose'][annotation.get('track_id')].append([keypoints[3], keypoints[4]])
                    frame_global_data['mask'][annotation.get('track_id')].append(keypoints[5])
                for i in range(section_range):
                    video_data = {
                        'obs_pose': defaultdict(list),
                        'future_pose': defaultdict(list),
                        'obs_mask': defaultdict(list),
                        'future_mask': defaultdict(list)
                    }
                    obs_pose_global = []
                    obs_mask_global = []
                    future_pose_global = []
                    future_mask_global = []
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        for pedestrian in frame_global_data['pose'].keys():
                            if frame_global_data['pose'][pedestrian].__len__() < j:
                                continue
                            if j <= self.obs_frame_num * self.skip_frame_num:
                                video_data['obs_pose'][pedestrian].append(frame_global_data['pose'][pedestrian][j - 1])
                                video_data['obs_mask'][pedestrian].append(frame_global_data['mask'][pedestrian][j - 1])
                            else:
                                video_data['future_pose'][pedestrian].append(
                                    frame_global_data['pose'][pedestrian][j - 1])
                                video_data['future_mask'][pedestrian].append(
                                    frame_global_data['mask'][pedestrian][j - 1])
                    for p_id in video_data['obs_pose'].keys():
                        if p_id in video_data['future_pose'].keys() and video_data['obs_pose'][
                            p_id].__len__() == self.obs_frame_num and \
                                video_data['future_pose'][
                                    p_id].__len__() == self.pred_frame_num:
                            obs_pose_global.append(video_data['obs_pose'][p_id])
                            obs_mask_global.append(video_data['obs_mask'][p_id])
                            future_pose_global.append(video_data['future_pose'][p_id])
                            future_mask_global.append(video_data['future_mask'][p_id])
                    obs_frames, future_frames = self.__generate_image_path(json_data, frame_global_data['frame_ids'],
                                                                           total_frame_num)
                    if not self.is_interactive:
                        for p_id in range(len(obs_pose_global)):
                            data.append(['%s-%d' % (video_id, i), obs_pose_global[p_id], future_pose_global[p_id],
                                         obs_mask_global[p_id], future_mask_global[p_id],
                                         obs_frames[p_id], future_frames[p_id]])
                    else:
                        data.append(
                            ['%s-%d' % (video_id, i), obs_pose_global, future_pose_global, obs_mask_global,
                             future_mask_global, obs_frames, future_frames])
                with open(os.path.join(self.output_dir, 'PoseTrack_global_{}.csv'.format(data_type)), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data)

    def disentangle_local(self, data_type='train'):
        header = ['video_section', 'observed_pose', 'future_pose', 'observed_mask', 'future_mask',
                  'obs_frames_related_path', 'future_frames_related_path']
        if self.custom_name:
            local_file_name = f'local_{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.csv'
        else:
            local_file_name = f'local_{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_PoseTrack.csv'
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
                json_data = json.load(json_file)
                annotations = json_data.get('annotations')
                if not annotations:
                    continue
                print('local: {}'.format(entry.name))
                video_id = json_data.get('images')[0].get('vid_id')
                frame_local_data = {
                    'pose': defaultdict(list),
                    'mask': defaultdict(list),
                    'frame_ids': defaultdict(list)
                }
                data = []
                for annotation in annotations:
                    frame_pose = []
                    frame_mask = []
                    keypoints = annotation.get('keypoints')
                    global_pose_x = keypoints[3]
                    global_pose_y = keypoints[4]
                    for i in range(1, len(keypoints) + 1):
                        if i % 3 == 0 and i != 6:
                            frame_mask.append(keypoints[i - 1])
                        elif i % 3 == 1 and i != 4:
                            frame_pose.append(keypoints[i - 1] - global_pose_x)
                        elif i % 3 == 2 and i != 5:
                            frame_pose.append(keypoints[i - 1] - global_pose_y)
                    frame_local_data['pose'][annotation.get('track_id')].append(frame_pose)
                    frame_local_data['mask'][annotation.get('track_id')].append(frame_mask)
                    frame_local_data['frame_ids'][annotation.get('track_id')].append(annotation.get('image_id'))
                for i in range(section_range):
                    video_data = {
                        'obs_pose': defaultdict(list),
                        'future_pose': defaultdict(list),
                        'obs_mask': defaultdict(list),
                        'future_mask': defaultdict(list)
                    }
                    obs_pose_local = []
                    obs_mask_local = []
                    future_pose_local = []
                    future_mask_local = []
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        for pedestrian in frame_local_data['pose'].keys():
                            if frame_local_data['pose'][pedestrian].__len__() < j:
                                continue
                            if j <= self.obs_frame_num * self.skip_frame_num:
                                video_data['obs_pose'][pedestrian].append(frame_local_data['pose'][pedestrian][j - 1])
                                video_data['obs_mask'][pedestrian].append(frame_local_data['mask'][pedestrian][j - 1])
                            else:
                                video_data['future_pose'][pedestrian].append(
                                    frame_local_data['pose'][pedestrian][j - 1])
                                video_data['future_mask'][pedestrian].append(
                                    frame_local_data['mask'][pedestrian][j - 1])
                    for p_id in video_data['obs_pose'].keys():
                        if p_id in video_data['future_pose'].keys() and video_data['obs_pose'][
                            p_id].__len__() == self.obs_frame_num and \
                                video_data['future_pose'][p_id].__len__() == self.pred_frame_num:
                            obs_pose_local.append(video_data['obs_pose'][p_id])
                            obs_mask_local.append(video_data['obs_mask'][p_id])
                            future_pose_local.append(video_data['future_pose'][p_id])
                            future_mask_local.append(video_data['future_mask'][p_id])
                    obs_frames, future_frames = self.__generate_image_path(json_data, frame_local_data['frame_ids'],
                                                                           total_frame_num)
                    if not self.is_interactive:
                        for p_id in range(len(obs_pose_local)):
                            data.append(['%s-%d' % (video_id, i), obs_pose_local[p_id], future_pose_local[p_id],
                                         obs_mask_local[p_id], future_mask_local[p_id],
                                         obs_frames[p_id], future_frames[p_id]])
                    else:
                        data.append(
                            ['%s%d' % (video_id, i), obs_pose_local, future_pose_local, obs_mask_local,
                             future_mask_local, obs_frames, future_frames])
                with open(os.path.join(self.output_dir, local_file_name), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data)
