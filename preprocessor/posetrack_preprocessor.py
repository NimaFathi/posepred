import csv
import json
import os
from collections import defaultdict

from preprocessor.preprocessor import Processor, OUTPUT_DIR


class PoseTrackPreprocessor(Processor):
    def __init__(self, mask, dataset_path, is_disentangle, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once):
        super(PoseTrackPreprocessor, self).__init__(dataset_path, is_disentangle, obs_frame_num, pred_frame_num,
                                                    skip_frame_num, use_video_once)
        self.dataset_total_frame_num = 30
        self.mask = mask

    def normal(self, data_type='train'):
        header = ['video_section', 'observed_pose', 'future_pose', 'observed_mask', 'future_mask']
        with open(os.path.join(OUTPUT_DIR, 'PoseTrack_{}.csv'.format(data_type)), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = 30 // (total_frame_num * 1) if self.use_video_once is False else 1

        for entry in os.scandir(self.dataset_path):
            with open(entry.path, 'r') as json_file:
                json_data = json.load(json_file)
                annotations = json_data.get('annotations')
                video_id = json_data.get('images')[0].get('vid_id')
                pose = defaultdict(list)
                mask = defaultdict(list)
                data = []
                for annotation in annotations:
                    frame_pose = []
                    frame_mask = []
                    keypoints = annotation.get('keypoints')
                    for i in range(1, len(keypoints) + 1):
                        if i % 3 == 0:
                            frame_mask.append(keypoints[i - 1])
                        frame_pose.append(keypoints[i - 1])
                    pose[annotation.get('track_id')].append(frame_pose)

                    mask[annotation.get('track_id')].append(frame_mask)
                for i in range(section_range):
                    video_dict = {
                        'obs_pose': defaultdict(list),
                        'future_pose': defaultdict(list),
                        'obs_mask': defaultdict(list),
                        'future_mask': defaultdict(list)
                    }
                    obs_pose = []
                    future_pose = []
                    obs_mask = []
                    future_mask = []
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        for pedestrian in pose.keys():
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
                    data.append(['%s_%d' % (video_id, i), obs_pose, future_pose, obs_mask, future_mask])
                with open(os.path.join(OUTPUT_DIR, 'PoseTrack_{}.csv'.format(data_type)), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data)

    def disentangle(self, data_type='train'):
        self.disentangle_global(data_type)
        self.disentangle_local(data_type)

    def disentangle_global(self, data_type='train'):
        header = ['video_section', 'observed_pose', 'future_pose', 'observed_mask', 'future_mask']
        with open(os.path.join(OUTPUT_DIR, 'PoseTrack_global_{}.csv'.format(data_type)), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.dataset_total_frame_num // (
                total_frame_num * self.skip_frame_num) if self.use_video_once is False else 1

        for entry in os.scandir(self.dataset_path):
            with open(entry.path, 'r') as json_file:
                json_data = json.load(json_file)
                annotations = json_data.get('annotations')
                video_id = json_data.get('images')[0].get('vid_id')
                global_pose = defaultdict(list)
                data = []
                for annotation in annotations:
                    keypoints = annotation.get('keypoints')
                    global_pose[annotation.get('track_id')].append([keypoints[3], keypoints[4]])
                for i in range(section_range):
                    obs_dict_global = defaultdict(list)
                    future_dict_global = defaultdict(list)
                    obs_global = []
                    future_global = []
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        for pedestrian in global_pose.keys():
                            if j <= self.obs_frame_num * self.skip_frame_num:
                                obs_dict_global[pedestrian].append(global_pose[pedestrian][j - 1])
                            else:
                                future_dict_global[pedestrian].append(global_pose[pedestrian][j - 1])
                    for p_id in obs_dict_global.keys():
                        if p_id in future_dict_global.keys() and obs_dict_global[
                            p_id].__len__() == self.obs_frame_num and \
                                future_dict_global[
                                    p_id].__len__() == self.pred_frame_num:
                            obs_global.append(obs_dict_global[p_id])
                            future_global.append(future_dict_global[p_id])
                    data.append(['%s_%d' % (video_id, i), obs_global, future_global])
                with open(os.path.join(OUTPUT_DIR, 'PoseTrack_global_{}.csv'.format(data_type)), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data)

    def disentangle_local(self, data_type='train'):
        header = ['video_section', 'observed_pose', 'future_pose', 'observed_mask', 'future_mask']
        with open(os.path.join(OUTPUT_DIR, 'PoseTrack_local_{}.csv'.format(data_type)), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.dataset_total_frame_num // (
                total_frame_num * self.skip_frame_num) if self.use_video_once is False else 1

        for entry in os.scandir(self.dataset_path):
            with open(entry.path, 'r') as json_file:
                json_data = json.load(json_file)
                annotations = json_data.get('annotations')
                video_id = json_data.get('images')[0].get('vid_id')
                frame_local_data = {
                    'pose': defaultdict(list),
                    'mask': defaultdict(list)
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
                    data.append(
                        ['%s_%d' % (video_id, i), obs_pose_local, future_pose_local, obs_mask_local, future_mask_local])
                with open(os.path.join(OUTPUT_DIR, 'PoseTrack_local_{}.csv'.format(data_type)), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data)
