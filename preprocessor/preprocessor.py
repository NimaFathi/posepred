class Processor:
    def __init__(self, is_interactive, is_disentangle, obs_frame_num, pred_frame_num, skip_frame_num, use_video_once):
        self.is_interactive = is_interactive
        self.is_disentangle = is_disentangle
        self.obs_frame_num = obs_frame_num
        self.pred_frame_num = pred_frame_num
        self.skip_frame_num = skip_frame_num
        self.use_video_once = use_video_once

    def store_processed_csv(self, dataset_path, dataset_name):
        # create csv files. (including train, valid, test)
        pass
