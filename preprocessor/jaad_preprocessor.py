import csv
import json
import logging
import os
import re
import shutil
from collections import namedtuple, defaultdict
from logging import config
from pathlib import Path

import numpy as np
import openpifpaf
from bs4 import BeautifulSoup
from scipy.interpolate import interp1d

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor

Rectangle = namedtuple('Rectangle', 'xtl ytl xbr ybr')

config.fileConfig('configs/logging.conf')
logger = logging.getLogger('consoleLogger')


class JAADPreprocessor(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name, annotate, image_dir, annotation_path):
        super(JAADPreprocessor, self).__init__(dataset_path, is_interactive, obs_frame_num,
                                               pred_frame_num, skip_frame_num, use_video_once, custom_name)

        self.output_dir = os.path.join(
            PREPROCESSED_DATA_DIR, 'JAAD_interactive') if self.is_interactive else os.path.join(
            PREPROCESSED_DATA_DIR, 'JAAD'
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.meta_data = {
            'avg_person': [],
            'count': 0,
            'sum2_pose': np.zeros(3),
            'sum_pose': np.zeros(3)
        }
        self.image_dir = image_dir
        self.annotate = annotate
        self.annotation_path = annotation_path

    def normal(self, data_type='train'):
        assert self.obs_frame_num > 0 and self.obs_frame_num is not None
        assert self.pred_frame_num > 0 and self.pred_frame_num is not None
        assert self.skip_frame_num >= 0 and self.skip_frame_num is not None
        assert self.dataset_path is not None
        logger.info('start creating JAAD normal static data ... ')
        header = [
            'video_section', 'observed_pose', 'future_pose',
            'observed_image_path', 'future_image_path'
        ]
        if self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.csv'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_JAAD.csv'
        with open(os.path.join(self.output_dir, output_file_name), 'w') as f_object:
            writer = csv.writer(f_object)
            writer.writerow(header)
        total_frame_num = (self.obs_frame_num + self.pred_frame_num) * (self.skip_frame_num + 1)
        frame_range = []
        joints_dir = self.__create_frame_poses()
        for entry in os.scandir(joints_dir):
            if not entry.name.endswith(".json"):
                continue
            with open(entry.path, 'r') as json_file:
                logger.info(f'file name: {entry.name}')
                video_name = re.search('(\w+).json', entry.name).group(1)
                data = json.load(json_file)
                if data == {}:
                    continue
                for ped_id, value in data.items():
                    for frame_number, pose in value.items():
                        frame_range.append(int(frame_number))
                min_frame = min(frame_range)
                max_frame = max(frame_range)
                section_range = (max_frame - min_frame + 1) // (
                        total_frame_num * (self.skip_frame_num + 1)
                ) if not self.use_video_once else 1
                for i in range(section_range):
                    obs_frame_range = [
                        i for i in range(
                            min_frame + i * total_frame_num,
                            min_frame + (i + 1) * total_frame_num - self.pred_frame_num * (self.skip_frame_num + 1),
                            self.skip_frame_num + 1
                        )
                    ]
                    pred_frame_range = [
                        i for i in range(
                            min_frame + (i + 1) * total_frame_num - self.pred_frame_num * (self.skip_frame_num + 1),
                            min_frame + (i + 1) * total_frame_num,
                            self.skip_frame_num + 1
                        )
                    ]
                    self.__create_data(data, obs_frame_range, pred_frame_range, video_name, output_file_name)
        shutil.rmtree(joints_dir)

    def __create_data(self, data, obs_frame_range, pred_frame_range, video_name, output_file_name):
        raw_data_obs = defaultdict(dict)
        raw_data_pred = defaultdict(dict)
        for ped_id, value in data.items():
            for frame_number, pose in value.items():
                if int(frame_number) in obs_frame_range:
                    raw_data_obs[ped_id][int(frame_number)] = pose
                elif int(frame_number) in pred_frame_range:
                    raw_data_pred[ped_id][int(frame_number)] = pose
        obs_poses = list()
        pred_poses = list()
        obs_image_path = list()
        pred_image_path = list()
        for ped_id in raw_data_obs.keys():
            if ped_id in raw_data_pred.keys() and len(raw_data_pred[ped_id]) > 1 and len(
                    raw_data_obs[ped_id]) > 1:
                obs_x = list(raw_data_obs[ped_id].keys())
                obs_y = np.array(list(raw_data_obs[ped_id].values()))
                obs_image_path.append(
                    [os.path.join(video_name, str(frame_number).zfill(5) + ".png") for
                     frame_number in obs_frame_range]
                )
                pred_image_path.append(
                    [os.path.join(video_name, str(frame_number).zfill(5) + ".png") for
                     frame_number in pred_frame_range]
                )
                obs_interp = interp1d(obs_x, obs_y, axis=0, fill_value="extrapolate")
                obs_poses.append(obs_interp(obs_frame_range).tolist())
                pred_x = list(raw_data_pred[ped_id].keys())
                pred_y = np.array(list(raw_data_pred[ped_id].values()))
                pred_interp = interp1d(pred_x, pred_y, axis=0, fill_value="extrapolate")
                pred_poses.append(pred_interp(pred_frame_range).tolist())
        with open(os.path.join(self.output_dir, output_file_name), 'a') as f_object:
            writer = csv.writer(f_object)
            if len(obs_poses) > 0:
                if self.is_interactive:
                    writer.writerow([video_name, obs_poses, pred_poses, obs_image_path, pred_image_path])
                else:
                    for i in range(len(obs_poses)):
                        writer.writerow(
                            [video_name, obs_poses[i], pred_poses[i], obs_image_path[i], pred_image_path[i]])

    def __create_frame_poses(self):
        if self.annotate is not False:
            assert self.image_dir is not None
            logger.warning(
                "It is better to create annotations yourself using openpifpaf with with desired methods and parameters")
            self.annotation_path = self.__create_annotations()
        else:
            assert self.annotation_path is not None
        for subdir, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if not file.endswith(".xml"):
                    continue
                with open(os.path.join(subdir, file), 'r') as xml_file:
                    video_name = "_".join(re.search('[\\/](\w+).xml', xml_file.name).groups())
                    correspondence_dict = defaultdict(dict)
                    xml_data = BeautifulSoup(xml_file.read(), 'lxml')
                    tracks = xml_data.find_all('track', {'label': 'pedestrian'})
                    ground_truth = defaultdict(dict)
                    for track in tracks:
                        bboxs = track.find_all('box')
                        p_id = bboxs[0].find('attribute', {'name': 'old_id'}).string
                        for bbox in bboxs:
                            if bbox.get('occluded') == "1":
                                continue
                            bbox_rec = Rectangle(
                                xtl=float(bbox.get('xtl')),
                                ytl=float(bbox.get('ytl')),
                                xbr=float(bbox.get('xbr')),
                                ybr=float(bbox.get('ybr'))
                            )
                            frame = bbox.get('frame')
                            ground_truth[frame][p_id] = bbox_rec
                    for frame in ground_truth.keys():
                        json_file_path = os.path.join(
                            self.annotation_path, video_name, str(frame).zfill(5) + ".png.predictions.json"
                        )
                        if not os.path.exists(json_file_path):
                            continue
                        with open(json_file_path, 'r') as json_file:
                            score_matrix = []
                            rectangles = []
                            data = json.load(json_file)
                            if not data:
                                continue
                            for pedestrian_data in data:
                                rectangles.append(Rectangle(
                                    xtl=pedestrian_data['bbox'][0],
                                    ytl=pedestrian_data['bbox'][1],
                                    xbr=pedestrian_data['bbox'][0] + pedestrian_data['bbox'][2],
                                    ybr=pedestrian_data['bbox'][1] + pedestrian_data['bbox'][3]
                                ))

                            for ped_id, ped_bbox in ground_truth[frame].items():
                                row_matrix = []
                                for i, rectangle in enumerate(rectangles):
                                    row_matrix.append(
                                        intersect_area(rectangle, ped_bbox) / (
                                                intersect_area(rectangle, rectangle) + intersect_area(
                                            ped_bbox, ped_bbox) - intersect_area(rectangle, ped_bbox)
                                        )
                                    )
                                score_matrix.append((ped_id, row_matrix))

                            for i in range(len(score_matrix)):
                                max_val = float('-inf')
                                max_index = 0
                                for j in range(len(score_matrix[i][1])):
                                    if score_matrix[i][1][j] > max_val:
                                        max_val = score_matrix[i][1][j]
                                        max_index = j
                                for k in range(len(score_matrix)):
                                    if len(score_matrix[k][1]) < max_index:
                                        score_matrix[k][1][max_index] = float('-inf')
                                if score_matrix[i][1][max_index] > 0.3:
                                    correspondence_dict[score_matrix[i][0]][frame] = [data[max_index]['keypoints'][v]
                                                                                      for v in range(
                                            len(data[max_index]['keypoints'])) if v % 3 != 2]
                    sorted_cor_dict = defaultdict(dict)
                    for i in sorted(correspondence_dict.keys()):
                        for j in sorted(correspondence_dict[i].keys()):
                            sorted_cor_dict[i][j] = correspondence_dict[i][j]
                    json_keypoints_dir = os.path.join(PREPROCESSED_DATA_DIR, 'JAAD', 'jsons')
                    path = Path(json_keypoints_dir)
                    path.mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(json_keypoints_dir, f'{video_name}.json'), 'w') as writer:
                        json.dump(sorted_cor_dict, writer, indent=4)
        return json_keypoints_dir

    def __create_annotations(self):
        logger.info("Create annotations using openpifpaf for JAAD in posepred")
        if os.path.exists(os.path.join(PREPROCESSED_DATA_DIR, 'openpifpaf/JAAD')):
            return PREPROCESSED_DATA_DIR + "openpifpaf/JAAD/"
        for subdir, dirs, files in os.walk(self.image_dir):
            annotation_dir = PREPROCESSED_DATA_DIR + 'openpifpaf/JAAD' + subdir.split(self.image_dir)[1]
            path = Path(annotation_dir)
            path.mkdir(parents=True, exist_ok=True)

            predictor = openpifpaf.Predictor(json_data=True)
            if files:
                new_files = []
                for i, filename in enumerate(files):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        new_files.append(subdir + "/" + filename)
                for pred, _, meta in predictor.images(new_files):
                    json_out_name = os.path.join(
                        annotation_dir,
                         os.path.basename(meta['file_name']) + ".predictions.json")
                    logger.debug('json output = %s', json_out_name)
                    with open(json_out_name, 'w') as f:
                        json.dump([ann for ann in pred], f, indent=4)
        return PREPROCESSED_DATA_DIR + "openpifpaf/JAAD/"


def intersect_area(a: Rectangle, b: Rectangle):
    dx = min(a.xbr, b.xbr) - max(a.xtl, b.xtl)
    dy = min(a.ybr, b.ybr) - max(a.ytl, b.ytl)
    if (dx > 0) and (dy > 0):
        return dx * dy
    else:
        return 0
