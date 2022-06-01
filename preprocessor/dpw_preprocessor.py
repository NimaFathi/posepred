from cmath import pi
import logging
import os
import re
from collections import defaultdict

import jsonlines
import numpy as np
import pandas as pd
import torch

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor

logger = logging.getLogger(__name__)


class Preprocessor3DPW(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name, save_total_frames, load_60Hz=False):
        super(Preprocessor3DPW, self).__init__(dataset_path, is_interactive, obs_frame_num,
                                               pred_frame_num, skip_frame_num, use_video_once,
                                               custom_name, save_total_frames)

        self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, '3DPW')
        if self.is_interactive:
            self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, '3DPW_interactive')
        elif self.save_total_frames:
            self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, '3DPW_total')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.meta_data = {
            'avg_person': [],
            'max_pose': np.zeros(3),
            'min_pose': np.array([1000.0, 1000.0, 1000.0]),
            'count': 0,
            'sum2_pose': np.zeros(3),
            'sum_pose': np.zeros(3)
        }

        self.load_60Hz = load_60Hz

    def normal(self, data_type='train'):
        logger.info('start creating 3DPW normal static data ... ')
        total_frame_num = self.obs_frame_num + self.pred_frame_num

        if self.save_total_frames:
            if self.custom_name:
                output_file_name = f'{data_type}_xyz_{self.custom_name}.jsonl'
            else:
                output_file_name = f'{data_type}_xyz_3dpw.jsonl'
        elif self.custom_name:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.jsonl'
        else:
            output_file_name = f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_3dpw.jsonl'
        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"

  
        self.dataset_path = os.path.join(self.dataset_path, data_type)

        for entry in os.scandir(self.dataset_path):
            if not entry.name.endswith('.pkl'):
                continue
            logger.info(f'file name: {entry.name}')
            pickle_obj = pd.read_pickle(entry.path)
            video_name = re.search('(\w+).pkl', entry.name).group(1)
            if self.load_60Hz:
                pose_data = np.array(pickle_obj['poses_60Hz'])
            else:
                pose_data = np.array(pickle_obj['jointPositions'])
                frame_ids_data = pickle_obj['img_frame_ids']
                cam_extrinsic = pickle_obj['cam_poses'][:, :3]
                cam_intrinsic = pickle_obj['cam_intrinsics'].tolist()

            pose_data = convertTo3D(pose_data)
            section_range = pose_data.shape[1] // (
                    total_frame_num * (self.skip_frame_num + 1)) if self.use_video_once is False else 1 

            if self.save_total_frames:
                section_range = 1
                total_frame_num = pose_data.shape[1]
                self.obs_frame_num = total_frame_num
                self.pred_frame_num = 0
                self.skip_frame_num = 0

            data = []
            for i in range(section_range):
                video_data = {
                    'obs_pose': defaultdict(list),
                    'future_pose': defaultdict(list),
                }
                if not self.load_60Hz:
                    video_data = {
                        'obs_pose': defaultdict(list),
                        'future_pose': defaultdict(list),
                        'obs_frames': defaultdict(list),
                        'future_frames': defaultdict(list),
                        'obs_cam_ext': list(),
                        'future_cam_ext': list()
                    }
                for j in range(1, total_frame_num * (self.skip_frame_num + 1) + 1, self.skip_frame_num + 1):
                    for p_id in range(pose_data.shape[0]):
                        if j <= (self.skip_frame_num + 1) * self.obs_frame_num:
                            video_data['obs_pose'][p_id].append(
                                pose_data[p_id, i * total_frame_num * (self.skip_frame_num + 1) + j - 1, :].tolist()
                            )
                            if not self.load_60Hz:
                                video_data['obs_frames'][p_id].append(
                                    f'{video_name}/image_{i * total_frame_num * (self.skip_frame_num + 1) + j - 1:05}.jpg'
                                )
                                if p_id == 0:
                                    video_data['obs_cam_ext'].append(
                                        cam_extrinsic[i * total_frame_num * (self.skip_frame_num + 1) + j - 1].tolist()
                                    )
                        else:
                            video_data['future_pose'][p_id].append(
                                pose_data[p_id, i * total_frame_num * (self.skip_frame_num + 1) + j - 1, :].tolist()
                            )
                            if not self.load_60Hz:
                                video_data['future_frames'][p_id].append(
                                    f'{video_name}/image_{i * total_frame_num * (self.skip_frame_num + 1) + j - 1:05}.jpg'
                                )
                                if p_id == 0:
                                    video_data['future_cam_ext'].append(
                                        cam_extrinsic[i * total_frame_num * (self.skip_frame_num + 1) + j - 1].tolist()
                                    )
                if len(list(video_data['obs_pose'].values())) > 0:
                    if data_type == 'train':
                        self.update_meta_data(self.meta_data, list(video_data['obs_pose'].values()), 3)
                    if not self.is_interactive:
                        for p_id in range(len(pose_data)):
                            data.append([
                                '%s-%d' % (video_name, i),
                                video_data['obs_pose'][p_id], video_data['future_pose'][p_id],
                                video_data['obs_frames'][p_id], video_data['future_frames'][p_id],
                                video_data['obs_cam_ext'], video_data['future_cam_ext'], cam_intrinsic
                            ] if not self.load_60Hz else [
                                '%s-%d' % (video_name, i),
                                video_data['obs_pose'][p_id], video_data['future_pose'][p_id]
                            ])
                    else:
                        data.append([
                            '%s-%d' % (video_name, i),
                            list(video_data['obs_pose'].values()), list(video_data['future_pose'].values()),
                            video_data['obs_frames'][0], video_data['future_frames'][0],
                            video_data['obs_cam_ext'], video_data['future_cam_ext'], cam_intrinsic
                        ] if not self.load_60Hz else [
                                 '%s-%d' % (video_name, i),
                            list(video_data['obs_pose'].values()), list(video_data['future_pose'].values()),
                            ])
            with jsonlines.open(os.path.join(self.output_dir, output_file_name), 'a') as writer:
                for data_row in data:
                    if not self.save_total_frames:
                        if not self.load_60Hz:
                            writer.write({
                                'video_section': data_row[0],
                                'observed_pose': data_row[1],
                                'future_pose': data_row[2],
                                'observed_image_path': data_row[3],
                                'future_image_path': data_row[4],
                                'observed_cam_extrinsic': data_row[5],
                                'future_cam_extrinsic': data_row[6],
                                'cam_intrinsic': data_row[7]
                            })
                        else:
                             writer.write({
                                'video_section': data_row[0],
                                'observed_pose': data_row[1],
                                'future_pose': data_row[2]
                            })
                    else:
                        if not self.load_60Hz:
                            writer.write({
                                'video_section': data_row[0],
                                'total_pose': data_row[1],
                                'total_image_path': data_row[3],
                                'total_cam_extrinsic': data_row[5],
                                'cam_intrinsic': data_row[7]
                            })
                        else:
                             writer.write({
                                'video_section': data_row[0],
                                'total_pose': data_row[1],
                            })

        self.save_meta_data(self.meta_data, self.output_dir, True, data_type)


p3d0_base = torch.tensor([[[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 7.2556e-02, -9.0371e-02, -4.9508e-03],
         [-7.0992e-02, -8.9911e-02, -4.2638e-03],
         [-2.9258e-03,  1.0815e-01, -2.7961e-02],
         [ 1.1066e-01, -4.7893e-01, -7.1666e-03],
         [-1.1376e-01, -4.8391e-01, -1.1530e-02],
         [ 3.5846e-03,  2.4726e-01, -2.5113e-02],
         [ 9.8395e-02, -8.8787e-01, -5.0576e-02],
         [-9.9592e-02, -8.9208e-01, -5.4003e-02],
         [ 5.3301e-03,  3.0330e-01, -1.3979e-04],
         [ 1.3125e-01, -9.4635e-01,  7.0107e-02],
         [-1.2920e-01, -9.4181e-01,  7.1206e-02],
         [ 2.4758e-03,  5.2506e-01, -3.7885e-02],
         [ 8.6329e-02,  4.2873e-01, -3.4415e-02],
         [-7.7794e-02,  4.2385e-01, -4.0395e-02],
         [ 8.1987e-03,  5.9696e-01,  1.8670e-02],
         [ 1.7923e-01,  4.6251e-01, -4.3923e-02],
         [-1.7389e-01,  4.5846e-01, -5.0048e-02],
         [ 4.4708e-01,  4.4718e-01, -7.2309e-02],
         [-4.3256e-01,  4.4320e-01, -7.3162e-02],
         [ 7.0520e-01,  4.5867e-01, -7.2730e-02],
         [-6.9369e-01,  4.5237e-01, -7.7453e-02]]])


def convertTo3D(pose_seq):
    res = []
    for pose in pose_seq:
        assert len(pose.shape) == 2 and pose.shape[1] == 72
        
        pose = torch.from_numpy(pose).float()
        pose = pose.view(-1, 72//3, 3)
        pose = pose[:, :-2]
        pose[:, 0] = 0
        res.append(ang2joint(pose).reshape(-1, 22 * 3).detach().numpy())
    return np.array(res)

def ang2joint(pose,
              parent={0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9,
                      15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}):
    """
    :param p3d0:[batch_size, joint_num, 3]
    :param pose:[batch_size, joint_num, 3]
    :param parent:
    :return:
    """

    assert len(pose.shape) == 3 and pose.shape[2] == 3 and p3d0_base.shape[1:] == pose.shape[1:]
    batch_num = pose.shape[0]
    p3d0 = p3d0_base.repeat([batch_num, 1, 1])

    jnum = 22

    J = p3d0
    R_cube_big = rodrigues(pose.contiguous().view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)
    results = []
    results.append(
        with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
    )

    for i in range(1, jnum):
        results.append(
            torch.matmul(
                results[parent[i]],
                with_zeros(
                    torch.cat(
                        (R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))),
                        dim=2
                    )
                )
            )
        )

    stacked = torch.stack(results, dim=1)
    J_transformed = stacked[:, :, :3, 3]
    return J_transformed


def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.
    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].
    Return:
    -------
    Rotation matrix of shape [batch_size * angle_num, 3, 3].
    """
    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
    # theta = torch.norm(r, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float).to(r.device)
    m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
         -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float).unsqueeze(dim=0) \
              + torch.zeros((theta_dim, 3, 3), dtype=torch.float)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R


def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.
    Parameter:
    ---------
    x: Tensor to be appended.
    Return:
    ------
    Tensor after appending of shape [4,4]
    """
    ones = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float
    ).expand(x.shape[0], -1, -1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret


def pack(x):
    """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.
    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]
    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.
    """
    zeros43 = torch.zeros(
        (x.shape[0], x.shape[1], 4, 3), dtype=torch.float).to(x.device)
    ret = torch.cat((zeros43, x), dim=3)
    return ret
