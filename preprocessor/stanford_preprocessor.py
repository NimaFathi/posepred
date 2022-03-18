import csv
import logging
import os
import zipfile
from glob import glob
from urllib.request import urlretrieve

import cdflib
import jsonlines
import numpy as np
import torch
from torch.autograd.variable import Variable
import copy

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor
from utils.others import expmap_to_quaternion, qfix, expmap_to_rotmat, expmap_to_euler

logger = logging.getLogger(__name__)

SPLIT = {
    'train': ['S1', 'S5', 'S6', 'S7', 'S8'],
    'validation': ['S1', 'S5', 'S6', 'S7', 'S8'],
    'test': ['S9', 'S11']
}


class StanfordPreprocessor(Processor):
    def __init__(self, dataset_path, is_interactive, skip_frame_num,
                 use_video_once, custom_name, save_total_frames):
        super(StanfordPreprocessor, self).__init__(dataset_path, is_interactive, 0,
                                                   0, skip_frame_num, use_video_once, custom_name, save_total_frames)
        assert self.is_interactive is False, 'human3.6m is not interactive'
        self.output_dir = os.path.join(
            PREPROCESSED_DATA_DIR, 'stanford36m_interactive') if self.is_interactive else os.path.join(
            PREPROCESSED_DATA_DIR, 'stanford36m'
        )
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
        self.subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

        self.acts = ["walking", "eating", "smoking", "discussion", "directions",
                     "greeting", "phoning", "posing", "purchases", "sitting",
                     "sittingdown", "takingphoto", "waiting", "walkingdog",
                     "walkingtogether"]

    def normal(self, data_type='train'):
        self.subjects = SPLIT[data_type]
        logger.info(
            'start creating Stanford3.6m preprocessed data from Stanford\'s Human3.6m dataset ... ')

        if self.save_total_frames:
            list_format = ["total"]
        else:
            list_format = ["xyz", "quaternion", "expmap", "rotmat", "euler"]

        for format_data in list_format:
            if self.custom_name:
                output_file_name = \
                    f'{data_type}_{format_data}_{self.custom_name}.jsonl'
            else:
                output_file_name = \
                    f'{data_type}_{format_data}_stanford3.6m.jsonl'

            assert os.path.exists(os.path.join(
                self.output_dir,
                output_file_name
            )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"
            

        for subject in self.subjects:
            logger.info("handling subject: {}".format(subject))
            for action in self.acts:
                expmap = self.expmap_rep(action, subject, data_type)
                positions = self.expmap2xyz_torch(torch.from_numpy(copy.deepcopy(expmap)).float())
                rotmat = self.rotmat_rep(action, subject, data_type)
                euler = self.euler_rep(action, subject, data_type)
                quat = self.quaternion_rep(action, subject, data_type)

                expmap = expmap.reshape(expmap.shape[0], -1)
                positions = positions.reshape(positions.shape[0], -1)
                rotmat = rotmat.reshape(rotmat.shape[0], -1)
                euler = euler.reshape(euler.shape[0], -1)
                quat = quat.reshape(quat.shape[0], -1)

                if self.save_total_frames == False:
                    self.obs_frame_num = 50
                    self.pred_frame_num = 25
                    total_frame_num = self.obs_frame_num + self.pred_frame_num
                    section_range = positions.shape[0] // (
                            total_frame_num * (self.skip_frame_num + 1)) if self.use_video_once is False else 1 # TODO: remove condition
                    
                    for i in range(section_range):
                        video_data = {
                            'observed_xyz_pose': list(),
                            'future_xyz_pose': list(),
                            'observed_quaternion_pose': list(),
                            'future_quaternion_pose': list(),
                            'observed_expmap_pose': list(),
                            'future_expmap_pose': list(),
                            'observed_rotmat_pose': list(),
                            'future_rotmat_pose': list(),
                            'observed_euler_pose': list(),
                            'future_euler_pose': list()
                        }
                        for j in range(0, total_frame_num * (self.skip_frame_num + 1), self.skip_frame_num + 1):
                            if j < (self.skip_frame_num + 1) * self.obs_frame_num:
                                video_data['observed_xyz_pose'].append(
                                    positions[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                                video_data['observed_quaternion_pose'].append(
                                    quat[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                                video_data['observed_expmap_pose'].append(
                                    expmap[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                                video_data['observed_rotmat_pose'].append(
                                    rotmat[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                                video_data['observed_euler_pose'].append(
                                    euler[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                            else:
                                video_data['future_xyz_pose'].append(
                                    positions[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                                video_data['future_quaternion_pose'].append(
                                    quat[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                                video_data['future_expmap_pose'].append(
                                    expmap[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                                video_data['future_rotmat_pose'].append(
                                    rotmat[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                                video_data['future_euler_pose'].append(
                                    euler[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())

                        for format_data in ["xyz", "quaternion", "expmap", "rotmat", "euler"]:
                            self.update_meta_data(self.meta_data, video_data['observed_xyz_pose'], 3)
                            if self.custom_name:
                                output_file_name = \
                                    f'{data_type}_{format_data}_{self.custom_name}.jsonl'
                            else:
                                output_file_name = \
                                    f'{data_type}_{format_data}_stanford3.6m.jsonl'
                            with jsonlines.open(os.path.join(self.output_dir, output_file_name), mode='a') as writer:
                                writer.write({
                                    'video_section': f'{subject}-{action}-{i}',
                                    'observed_pose': video_data[f'observed_{format_data}_pose'],
                                    'future_pose': video_data[f'future_{format_data}_pose'],
                                    'action': action
                                })
                else:
                    video_data = {
                        'xyz_pose': positions.tolist()[::self.skip_frame_num + 1],
                        'quaternion_pose': quat.tolist()[::self.skip_frame_num + 1],
                        'expmap_pose': expmap.tolist()[::self.skip_frame_num + 1],
                        'rotmat_pose': rotmat.tolist()[::self.skip_frame_num + 1],
                        'euler_pose': euler.tolist()[::self.skip_frame_num + 1],
                        'action': action
                    }

                    if self.custom_name:
                        output_file_name = \
                            f'{data_type}_total_{self.custom_name}.jsonl'
                    else:
                        output_file_name = \
                            f'{data_type}_total_stanford3.6m.jsonl'


                    self.update_meta_data(self.meta_data, video_data['xyz_pose'], 3)
                    with jsonlines.open(os.path.join(self.output_dir, output_file_name), mode='a') as writer:
                        writer.write({
                            'video_section': f'{subject}-{action}',
                            'action': f'{action}',
                            'xyz_pose': video_data['xyz_pose'],
                            'quaternion_pose': video_data['quaternion_pose'],
                            'expmap_pose': video_data['expmap_pose'],
                            'rotmat_pose': video_data['rotmat_pose'],
                            'euler_pose': video_data['euler_pose']
                        })
        self.save_meta_data(self.meta_data, self.output_dir, True, data_type)

    def expmap_rep(self, action, subject, data_type):
        data = self.__read_file(action, self.dataset_path, subject, data_type)
        return data

    def rotmat_rep(self, action, subject, data_type):
        data = self.expmap_rep(action, subject, data_type)
        data = data.reshape(data.shape[0], -1, 3)[:, 1:]
        data = expmap_to_rotmat(data)
        return data

    def euler_rep(self, action, subject, data_type):
        data = self.expmap_rep(action, subject, data_type)
        data = data.reshape(data.shape[0], -1, 3)[:, 1:]
        data = expmap_to_euler(data)
        return data

    def quaternion_rep(self, action, subject, data_type):
        data = self.expmap_rep(action, subject, data_type)
        data = data.reshape(data.shape[0], -1, 3)[:, 1:]
        quat = expmap_to_quaternion(-data)
        quat = qfix(quat)
        quat = quat.reshape(-1, 32 * 4)
        return quat.reshape(-1, 32 * 4)

    def expmap2xyz_torch(self, expmap):
        """
        convert expmaps to joint locations
        :param expmap: N*99
        :return: N*32*3
        """
        expmap[:, 0:6] = 0
        parent, offset, rotInd, expmapInd = self._some_variables()
        xyz = self.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
        return xyz

    def _some_variables(self):
        """
        borrowed from
        https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L100
        We define some variables that are useful to run the kinematic tree
        Args
          None
        Returns
          parent: 32-long vector with parent-child relationships in the kinematic tree
          offset: 96-long vector with bone lenghts
          rotInd: 32-long list with indices into angles
          expmapInd: 32-long list with indices into expmap angles
        """

        parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                           17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

        offset = np.array(
            [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
             -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
             0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
             0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
             257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
             0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
             0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
             0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
             0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
             0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
        offset = offset.reshape(-1, 3)

        rotInd = [[5, 6, 4],
                  [8, 9, 7],
                  [11, 12, 10],
                  [14, 15, 13],
                  [17, 18, 16],
                  [],
                  [20, 21, 19],
                  [23, 24, 22],
                  [26, 27, 25],
                  [29, 30, 28],
                  [],
                  [32, 33, 31],
                  [35, 36, 34],
                  [38, 39, 37],
                  [41, 42, 40],
                  [],
                  [44, 45, 43],
                  [47, 48, 46],
                  [50, 51, 49],
                  [53, 54, 52],
                  [56, 57, 55],
                  [],
                  [59, 60, 58],
                  [],
                  [62, 63, 61],
                  [65, 66, 64],
                  [68, 69, 67],
                  [71, 72, 70],
                  [74, 75, 73],
                  [],
                  [77, 78, 76],
                  []]

        expmapInd = np.split(np.arange(4, 100) - 1, 32)

        return parent, offset, rotInd, expmapInd

    def fkl_torch(self, angles, parent, offset, rotInd, expmapInd):
        """
        pytorch version of fkl.
        convert joint angles to joint locations
        batch pytorch version of the fkl() method above
        :param angles: N*99
        :param parent:
        :param offset:
        :param rotInd:
        :param expmapInd:
        :return: N*joint_n*3
        """
        n_a = angles.data.shape[0]
        j_n = offset.shape[0]
        p3d = Variable(torch.from_numpy(offset)).float().unsqueeze(0).repeat(n_a, 1, 1)
        angles = angles[:, 3:].contiguous().view(-1, 3)

        theta = torch.norm(angles, 2, 1)
        r0 = torch.div(angles, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
        r1 = torch.zeros_like(r0).repeat(1, 3)
        r1[:, 1] = -r0[:, 2]
        r1[:, 2] = r0[:, 1]
        r1[:, 5] = -r0[:, 0]
        r1 = r1.view(-1, 3, 3)
        r1 = r1 - r1.transpose(1, 2)
        n = r1.data.shape[0]
        R = (torch.eye(3, 3).repeat(n, 1, 1).float() + torch.mul(
            torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
            (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))).view(n_a, j_n, 3, 3)

        for i in np.arange(1, j_n):
            if parent[i] > 0:
                R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :]).clone()
                p3d[:, i, :] = torch.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :]
        return p3d

    @staticmethod
    def __read_file(action, rot_dir_path, subject, data_type):
        '''
        Read an individual file in expmap format,
        and return a NumPy tensor with shape (sequence length, number of joints, 3).
        '''
        action = action.replace('WalkTogether', 'WalkingTogether').replace(
            'WalkDog', 'WalkingDog')
        if action.lower().__contains__('photo'):
            action = 'TakingPhoto'
        action_number = 1 if len(action.split(" ")) == 2 else 2
        path_to_read = os.path.join(rot_dir_path, 'dataset', subject,
                                    f'{action.split(" ")[0].lower()}_{action_number}.txt')
        data = []
        with open(path_to_read, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                data.append(row)
        data = np.array(data, dtype='float64')
        if data_type == 'train':
            data = data[:95 * data.shape[0] // 100]
        elif data_type == 'validation':
            data = data[95 * data.shape[0] // 100:]
        return data

    @staticmethod
    def delete_redundant_files():
        output_directory = os.path.join(PREPROCESSED_DATA_DIR, 'H3.6m_rotations')
        h36_folder = os.path.join(output_directory, 'h3.6m')
        os.remove(h36_folder + ".zip")

