import json
import logging
import os
from collections import namedtuple, defaultdict
from pathlib import Path

import numpy as np
import openpifpaf
from bs4 import BeautifulSoup
from openpifpaf.predict import out_name

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor

Rectangle = namedtuple('Rectangle', 'xtl ytl xbr ybr')

logger = logging.getLogger(__name__)


class PIEPreprocessor(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name, annotate, image_dir, annotation_path):
        super(PIEPreprocessor, self).__init__(dataset_path, is_interactive, obs_frame_num,
                                              pred_frame_num, skip_frame_num, use_video_once, custom_name)

        self.output_dir = os.path.join(
            PREPROCESSED_DATA_DIR, 'PIE_interactive') if self.is_interactive else os.path.join(
            PREPROCESSED_DATA_DIR, 'PIE'
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
        correspondence_dict = defaultdict(dict)
        counter = 0
        ground_truth_ped_count = 0
        pred_ped_count = 0
        ious = []
        if self.annotate is not False:
            assert self.image_dir is not None
            self.annotation_path = self.create_annotations()
        else:
            assert self.annotation_path is not None
        for subdir, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if not file.endswith(".xml"):
                    continue
                with open(os.path.join(subdir, file), 'r') as xml_file:
                    xml_data = BeautifulSoup(xml_file.read(), 'lxml')
                    tracks = xml_data.find_all('track', {'label': 'pedestrian'})
                    ground_truth = defaultdict(dict)
                    for track in tracks:
                        bboxs = track.find_all('box')
                        p_id = bboxs[0].find('attribute', {'name': 'id'}).string
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
                            self.annotation_path,
                            str(frame) + ".png.predictions.json"
                        )
                        ground_truth_ped_count += ground_truth[frame].__len__()
                        if not os.path.exists(json_file_path):
                            continue
                        with open(json_file_path, 'r') as json_file:
                            score_matrix = []
                            intersection_matrix = []
                            rectangles = []
                            data = json.load(json_file)
                            if not data:
                                continue
                            pred_ped_count += len(data)
                            for pedestrian_data in data:
                                rectangles.append(Rectangle(
                                    xtl=pedestrian_data['bbox'][0],
                                    ytl=pedestrian_data['bbox'][1],
                                    xbr=pedestrian_data['bbox'][0] + pedestrian_data['bbox'][2],
                                    ybr=pedestrian_data['bbox'][1] + pedestrian_data['bbox'][3]
                                ))

                            for ped_id, ped_bbox in ground_truth[frame].items():
                                row_matrix = []
                                row_intersection = []
                                for i, rectangle in enumerate(rectangles):
                                    row_intersection.append(intersect_area(rectangle, ped_bbox))
                                    row_matrix.append(
                                        intersect_area(rectangle, ped_bbox) +
                                        keypoints_overlap_score(data[i]['keypoints'], ped_bbox)
                                    )
                                intersection_matrix.append(row_intersection)
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
                                if score_matrix[i][1][max_index] > 1:
                                    ious.append(
                                        intersection_matrix[i][max_index] /
                                        (
                                                calculate_are(ground_truth[frame][score_matrix[i][0]]) +
                                                calculate_are(rectangles[max_index]) -
                                                intersection_matrix[i][max_index]
                                        )
                                    )
                                    correspondence_dict[score_matrix[i][0]][frame] = data[max_index]['keypoints']
                                else:
                                    counter += 1
        print(
            f'IOU percentage: {sum(ious) / len(ious)}\n'
            f'openpifpaf pred pedestrian count: {pred_ped_count}\n'
            f'ground truth pedestrian count: {ground_truth_ped_count}\n'
            f'number of matched ious: {len(ious)}\n'
            f'number of unmatched ious: {counter}'
        )
        sorted_cor_dict = defaultdict(dict)
        for i in sorted(correspondence_dict.keys()):
            for j in sorted(correspondence_dict[i].keys()):
                sorted_cor_dict[i][j] = correspondence_dict[i][j]

        with open(f'{PREPROCESSED_DATA_DIR}/PIE.json', 'w') as json_file:
            json.dump(sorted_cor_dict, json_file, indent=4)

    def create_annotations(self):

        if os.path.exists(os.path.join(PREPROCESSED_DATA_DIR, 'openpifpaf/PIE')):
            return PREPROCESSED_DATA_DIR + "openpifpaf/PIE/"
        annotation_dir = PREPROCESSED_DATA_DIR + 'openpifpaf/PIE'
        path = Path(annotation_dir)
        path.mkdir(parents=True, exist_ok=True)
        for subdir, dirs, files in os.walk(self.image_dir):
            predictor = openpifpaf.Predictor(json_data=True)
            if files:
                new_files = []
                for i, filename in enumerate(files):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        new_files.append(subdir + "/" + filename)
                for pred, _, meta in predictor.images(new_files):
                    json_out_name = out_name(
                        annotation_dir, meta['file_name'], '.predictions.json')
                    logger.debug('json output = %s', json_out_name)
                    with open(json_out_name, 'w') as f:
                        json.dump([ann for ann in pred], f, indent=4)
        return PREPROCESSED_DATA_DIR + "openpifpaf/PIE/"


def intersect_area(a: Rectangle, b: Rectangle):
    dx = min(a.xbr, b.xbr) - max(a.xtl, b.xtl)
    dy = min(a.ybr, b.ybr) - max(a.ytl, b.ytl)
    if (dx > 0) and (dy > 0):
        return dx * dy
    else:
        return 0


def calculate_are(a: Rectangle):
    return (a.xbr - a.xtl) * (a.ybr - a.ytl)


def keypoints_overlap_score(keypoints, bbox: Rectangle):
    assert len(keypoints) == 51
    overlap_num = 0
    for i in range(len(keypoints) // 3):
        x, y = keypoints[3 * i], keypoints[3 * i + 1]
        if bbox.xtl <= x <= bbox.xbr and bbox.ytl <= y <= bbox.ybr:
            overlap_num += 1
    return overlap_num / len(keypoints)
