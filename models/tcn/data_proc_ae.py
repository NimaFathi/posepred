import torch
from torch import nn
import numpy as np

import sys
# sys.path.append('/home/zahra/workshop/')
# sys.path.append('/home/zahra/workshop/h36m_exp/')
# from posepred.utils.save_load import load_snapshot
# from viz import *

def joint_to_index(x):
    return np.concatenate((x * 3, x * 3 + 1, x * 3 + 2))


dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

index_to_ignore = np.array([16, 20, 23, 24, 28, 31])
index_to_ignore = joint_to_index(index_to_ignore)

index_to_equal = np.array([13, 19, 22, 13, 27, 30])
index_to_equal = joint_to_index(index_to_equal)

index_to_copy = np.array([0, 1, 6, 11])
index_to_copy = joint_to_index(index_to_copy)


class Preprocess(nn.Module):
    def __init__(self, args, encoder):
        super(Preprocess, self).__init__()
        self.args = args
        # ae = load_snapshot(args.ae_path, model_only=True)
        self.encoder = encoder
        self.encoder = self.encoder.to(args.device)

    def forward(self, observed_pose):
        return self.encoder(observed_pose) #observed_pose[:, :, dim_used]


class Postprocess(nn.Module):
    def __init__(self, args, decoder):
        super(Postprocess, self).__init__()
        self.args = args
        # ae = load_snapshot(args.ae_path, model_only=True)
        self.decoder = decoder
        self.decoder = self.decoder.to(args.device)

    def forward(self, observed_pose, pred_pose):
        # x = torch.zeros([pred_pose.shape[0], pred_pose.shape[1], 96]).to(self.args.device)
        # x[:, :, dim_used] = pred_pose
        # x[:, :, index_to_copy] = observed_pose[:, -1:, index_to_copy]
        # x[:, :, index_to_ignore] = x[:, :, index_to_equal]
        return self.decoder(pred_pose)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    import jsonlines
    args = Namespace(device='cuda', ae_path='/home/zahra/models/pose_ae/linear_24.pt')
    preproc = Preprocess(args).to(args.device)
    postproc = Postprocess(args).to(args.device)

    path = '/home/zahra/workshop/posepred/preprocessed_data/stanford36m/train_total_xyz.jsonl'
    sequences = []
    with jsonlines.open(path) as reader:
        i = 0
        for seq in reader:
            sequences.append(np.array(seq['xyz_pose']))
            i += 1
            if i == 100: break

    for i in [21]:
        sequence_gt = sequences[i] # T, 96
        inputs = torch.tensor(sequence_gt[:1]).unsqueeze(0).to(args.device).float() # 1, 10, 96
        spherical = preproc(inputs)
        outputs = postproc(inputs, spherical)
        # print(i, torch.allclose(inputs, outputs)) #, atol=1e-4, rtol=1e-4))
        inputs = inputs.squeeze(0).cpu().detach().numpy()
        outputs = outputs.squeeze(0).cpu().detach().numpy()
        print(inputs.shape, outputs.shape)
        # visualize(inputs, outputs, 25, 'ae.gif')
        for i in range(32):
            inx = inputs.reshape(32, 3)[i]
            outx = outputs.reshape(32, 3)[i]
            print(i, np.allclose(inx, outx, atol=1e-4, rtol=1e-4), inx, outx)