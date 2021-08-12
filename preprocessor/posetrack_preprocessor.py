import csv
import json
import os
from collections import defaultdict

from preprocessor.preprocessor import Processor, OUTPUT_DIR


class PoseTrackPreprocessor(Processor):
    def __init__(self, mask, is_disentangle, obs_frame_num, pred_frame_num, skip_frame_num, use_video_once):
        super(PoseTrackPreprocessor, self).__init__(is_disentangle, obs_frame_num, pred_frame_num, skip_frame_num,
                                                    use_video_once)
        self.dataset_total_frame_num = 30
        self.mask = mask

    def normal(self, data_type='train'):
        header = ['video_section', 'observed_pose', 'future_pose']
        with open(os.path.join(OUTPUT_DIR, 'PoseTrack_{}.csv'.format(data_type)), 'w') as f_object:
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
                pose = defaultdict(list)
                data = []
                for annotation in annotations:
                    frame_pose = []
                    keypoints = annotation.get('keypoints')
                    for i in range(1, len(keypoints) + 1):
                        if i % 3 == 0:
                            continue
                        frame_pose.append(keypoints[i - 1])
                    pose[annotation.get('track_id')].append(frame_pose)
                for i in range(section_range):
                    obs_dict = defaultdict(list)
                    future_dict = defaultdict(list)
                    obs = []
                    future = []
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        for pedestrian in pose.keys():
                            if j <= self.obs_frame_num * self.skip_frame_num:
                                obs_dict[pedestrian].append(pose[pedestrian][j - 1])
                            else:
                                future_dict[pedestrian].append(pose[pedestrian][j - 1])
                    for p_id in obs_dict.keys():
                        if p_id in future_dict.keys() and obs_dict[p_id].__len__() == self.obs_frame_num and \
                                future_dict[
                                    p_id].__len__() == self.pred_frame_num:
                            obs.append(obs_dict[p_id])
                            future.append(future_dict[p_id])
                    data.append(['%s_%d' % (video_id, i), obs, future])
                with open(os.path.join(OUTPUT_DIR, 'PoseTrack_{}.csv'.format(data_type)), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data)

    def disentangle_global(self, data_type='train'):
        header = ['video_section', 'observed_pose', 'future_pose']
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
        header = ['video_section', 'observed_pose', 'future_pose']
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
                local_pose = defaultdict(list)
                data = []
                for annotation in annotations:
                    frame_pose = []
                    keypoints = annotation.get('keypoints')
                    global_pose_x = keypoints[3]
                    global_pose_y = keypoints[4]
                    for i in range(1, len(keypoints) + 1):
                        if i % 3 == 0:
                            continue
                        elif i % 3 == 1 and i != 4:
                            frame_pose.append(keypoints[i - 1] - global_pose_x)
                        elif i % 3 == 2 and i != 5:
                            frame_pose.append(keypoints[i - 1] - global_pose_y)
                    local_pose[annotation.get('track_id')].append(frame_pose)
                for i in range(section_range):
                    obs_dict_local = defaultdict(list)
                    future_dict_local = defaultdict(list)
                    obs_local = []
                    future_local = []
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        for pedestrian in local_pose.keys():
                            if j <= self.obs_frame_num * self.skip_frame_num:
                                obs_dict_local[pedestrian].append(local_pose[pedestrian][j - 1])
                            else:
                                future_dict_local[pedestrian].append(local_pose[pedestrian][j - 1])
                    for p_id in obs_dict_local.keys():
                        if p_id in future_dict_local.keys() and obs_dict_local[p_id].__len__() == self.obs_frame_num and \
                                future_dict_local[
                                    p_id].__len__() == self.pred_frame_num:
                            obs_local.append(obs_dict_local[p_id])
                            future_local.append(future_dict_local[p_id])
                    data.append(['%s_%d' % (video_id, i), obs_local, future_local])
                with open(os.path.join(OUTPUT_DIR, 'PoseTrack_local_{}.csv'.format(data_type)), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data)

    def mask(self, data_type='train'):
        header = ['video_section', 'observed_mask', 'future_mask']
        with open(os.path.join(OUTPUT_DIR, 'PoseTrack_{}.csv'.format(data_type)), 'w') as f_object:
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
                pose = defaultdict(list)
                data = []
                for annotation in annotations:
                    frame_pose = []
                    keypoints = annotation.get('keypoints')
                    for i in range(1, len(keypoints) + 1):
                        if i % 3 != 0:
                            continue
                        frame_pose.append(keypoints[i - 1])
                    pose[annotation.get('track_id')].append(frame_pose)
                for i in range(section_range):
                    obs_dict = defaultdict(list)
                    future_dict = defaultdict(list)
                    obs = []
                    future = []
                    for j in range(1, total_frame_num * self.skip_frame_num + 1, self.skip_frame_num):
                        for pedestrian in pose.keys():
                            if j <= self.obs_frame_num * self.skip_frame_num:
                                obs_dict[pedestrian].append(pose[pedestrian][j - 1])
                            else:
                                future_dict[pedestrian].append(pose[pedestrian][j - 1])
                    for p_id in obs_dict.keys():
                        if p_id in future_dict.keys() and obs_dict[p_id].__len__() == self.obs_frame_num and \
                                future_dict[
                                    p_id].__len__() == self.pred_frame_num:
                            obs.append(obs_dict[p_id])
                            future.append(future_dict[p_id])
                    data.append(['%s_%d' % (video_id, i), obs, future])
                with open(os.path.join(OUTPUT_DIR, 'PoseTrack_{}.csv'.format(data_type)), 'a') as f_object:
                    writer = csv.writer(f_object)
                    writer.writerows(data)
