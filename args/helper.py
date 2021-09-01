from json import JSONEncoder
import numpy as np


class TrainerArgs:
    def __init__(self, epochs, is_interactive, start_epoch, lr, decay_factor, decay_patience, distance_loss,
                 mask_loss_weight, snapshot_interval):
        self.epochs = epochs
        self.is_interactive = is_interactive
        self.start_epoch = start_epoch
        self.lr = lr
        self.decay_factor = decay_factor
        self.decay_patience = decay_patience
        self.distance_loss = distance_loss
        self.mask_loss_weight = mask_loss_weight
        self.snapshot_interval = snapshot_interval
        self.save_dir = None


class DataloaderArgs:
    def __init__(self, dataset_name, keypoint_dim, is_interactive, persons_num, use_mask, skip_frame, batch_size,
                 shuffle, pin_memory, num_workers, is_testing=False, is_visualizing=False):
        self.dataset_name = dataset_name
        self.keypoint_dim = keypoint_dim
        self.is_interactive = is_interactive
        self.persons_num = persons_num
        self.use_mask = use_mask
        self.skip_frame = skip_frame
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.is_testing = is_testing
        self.is_visualizing = is_visualizing


class ModelArgs:
    def __init__(self, model_name, use_mask, keypoint_dim, hidden_size=200, hardtanh_limit=10, n_layers=1,
                 dropout_enc=0, dropout_pose_dec=0, dropout_mask_dec=0):
        self.model_name = model_name
        self.use_mask = use_mask
        self.keypoint_dim = keypoint_dim
        self.hidden_size = hidden_size
        self.hardtanh_limit = hardtanh_limit
        self.n_layers = n_layers
        self.dropout_enc = dropout_enc
        self.dropout_pose_dec = dropout_pose_dec
        self.dropout_mask_dec = dropout_mask_dec
        self.pred_frames_num = None
        self.keypoints_num = None


class JSONEncoder_(JSONEncoder):
    def default(self, o):
        return o.__dict__


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
