import json
import logging
import os
from pathlib import Path

import numpy as np
import openpifpaf
from openpifpaf.predict import out_name

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor

logger = logging.getLogger(__name__)


class PreprocessorPIE(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name):
        super(PreprocessorPIE, self).__init__(dataset_path, is_interactive, obs_frame_num,
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

    def normal(self, data_type='train'):
        self.create_annotations()

    def create_annotations(self):
        for subdir, dirs, files in os.walk(self.dataset_path):
            annotation_dir = PREPROCESSED_DATA_DIR + 'openpifpaf/PIE' + subdir.split(self.dataset_path)[1]
            path = Path(annotation_dir)
            path.mkdir(parents=True, exist_ok=True)
            predictor = openpifpaf.Predictor(json_data=True)
            if files:
                new_files = []
                for i, filename in enumerate(files):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                        new_files.append(subdir + "/" + filename)
                for pred, _, meta in predictor.images(new_files):
                    # json output
                    json_out_name = out_name(
                        annotation_dir, meta['file_name'], '.predictions.json')
                    logger.debug('json output = %s', json_out_name)
                    with open(json_out_name, 'w') as f:
                        json.dump([ann for ann in pred], f, indent=4)
        self.dataset_path = PREPROCESSED_DATA_DIR + "openpifpaf/PIE/"


if __name__ == '__main__':
    s = PreprocessorPIE('home/nima', 16, 14, 0, True, True, None)
    s.create_annotations('/home/nima/EPFL/images')
