import os
from consts import ROOT_DIR

OUTPUT_DIR = os.path.join(ROOT_DIR, 'preprocessed_data/')


class Processor:
    def __init__(self, dataset_path, is_disentangle, obs_frame_num, pred_frame_num, skip_frame_num, use_video_once):
        self.is_disentangle = is_disentangle
        self.obs_frame_num = obs_frame_num
        self.pred_frame_num = pred_frame_num
        self.skip_frame_num = skip_frame_num
        self.use_video_once = use_video_once
        self.dataset_path = dataset_path

    def normal(self, data_type='train'):
        pass

    def disentangle(self, data_type='train'):
        pass
