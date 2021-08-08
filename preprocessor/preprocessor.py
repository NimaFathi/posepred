import json
import os
from collections import defaultdict
from consts import ROOT_DIR
import numpy as np
import csv


OUTPUT_DIR = os.path.join(ROOT_DIR, 'preprocessor')


class Processor:
    def __init__(self, is_interactive, is_disentangle, obs_frame_num, pred_frame_num, skip_frame_num, use_video_once):
        self.is_interactive = is_interactive
        self.is_disentangle = is_disentangle
        self.obs_frame_num = obs_frame_num
        self.pred_frame_num = pred_frame_num
        self.skip_frame_num = skip_frame_num
        self.use_video_once = use_video_once


