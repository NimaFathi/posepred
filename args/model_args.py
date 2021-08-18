import argparse


class ModelArgs:
    def __init__(self, model_name, hidden_size=200, hardtanh_limit=10, n_layers=1, dropout_enc=0,
                 dropout_pose_dec=0, dropout_mask_dec=0):
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.hardtanh_limit = hardtanh_limit
        self.n_layers = n_layers
        self.dropout_enc = dropout_enc
        self.dropout_pose_dec = dropout_pose_dec
        self.dropout_mask_dec = dropout_mask_dec
        self.pred_frames_num = None
        self.keypoint_dim = None
        self.keypoints_num = None
        self.use_mask = None


def parse_model_args():
    parser = argparse.ArgumentParser('Argument for Model')
    parser.add_argument('--model_name', type=str, help='model_name')
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--hardtanh_limit', type=float, default=10)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dropout_enc', type=float, default=0)
    parser.add_argument('--dropout_pose_dec', type=float, default=0)
    parser.add_argument('--dropout_mask_dec', type=float, default=0)
    model_args = parser.parse_args()
    return model_args
