import csv
import json
import os
from collections import defaultdict

from preprocessor.preprocessor import Processor, OUTPUT_DIR


class PoseTrack_Preprocessor(Processor):
    def __init__(self, mask):
        super(PoseTrack_Preprocessor, self).__init__()
        self.frame_num = 30
        self.mask = mask

    def normal(self, data_type='train'):
        header = ['video_section', 'observed_pose', 'future_pose']
        with open(os.path.join(OUTPUT_DIR, 'PoseTrack_{}.csv'.format(data_type)), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.frame_num // (total_frame_num * self.skip_frame_num) if self.use_video_once is False else 1

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
                    for i in range(len(keypoints)):
                        if i % 3 == 0:
                            continue
                        frame_pose.append(keypoints[i])
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

    def mask(self, data_type='train'):
        header = ['video_section', 'observed_mask', 'future_mask']
        with open(os.path.join(OUTPUT_DIR, 'PoseTrack_{}.csv'.format(data_type)), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        total_frame_num = self.obs_frame_num + self.pred_frame_num
        section_range = self.frame_num // (total_frame_num * self.skip_frame_num) if self.use_video_once is False else 1

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
                    for i in range(len(keypoints)):
                        if i % 3 != 0:
                            continue
                        frame_pose.append(keypoints[i])
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
