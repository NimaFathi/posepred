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


class PreprocessorOur(Processor):
    def __init__(self, dataset_path, is_interactive, skip_frame_num,
                 use_video_once, custom_name):
        super(PreprocessorOur, self).__init__(dataset_path, is_interactive, 0,
                                              0, skip_frame_num, use_video_once, custom_name)
        assert self.is_interactive is False, 'human3.6m is not interactive'
        self.output_dir = os.path.join(
            PREPROCESSED_DATA_DIR, 'human36m_interactive') if self.is_interactive else os.path.join(
            PREPROCESSED_DATA_DIR, 'human36m'
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
            'start creating Human3.6m normal static data \
                    from original Human3.6m dataset (CDF files) ... ')
        if self.custom_name:
            output_file_name = \
                f'{data_type}_{self.skip_frame_num}_{self.custom_name}.jsonl'
        else:
            output_file_name = \
                f'{data_type}_{self.skip_frame_num}_human3.6m.jsonl'
        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"
        for subject in self.subjects:
            logger.info("handling subject: {}".format(subject))
            # subject_pose_path = os.path.join(self.dataset_path, subject, 'MyPoseFeatures/D3_Positions/*.cdf')
            # file_list_pose = glob(subject_pose_path)
            # assert len(file_list_pose) == 30, "Expected 30 files for subject " + subject + ", got " + str(
                # len(file_list_pose))
            for action in self.acts:
                # action = os.path.splitext(os.path.basename(f))[0]
                if subject == 'S11' and action == 'Directions':
                    continue  # Discard corrupted video
                canonical_name = action.replace('TakingPhoto', 'Photo') \
                    .replace('WalkingDog', 'WalkDog')
                # hf = cdflib.CDF(f)
                # positions = hf['xyz_pose'].reshape(-1, 96)
                # if data_type == 'train':
                #     positions = positions[:95 * positions.shape[0] // 100]
                # elif data_type == 'validation':
                #     positions = positions[95 * positions.shape[0] // 100:]
                # positions /= 1000
                expmap = self.expmap_rep(action, subject, data_type)
                positions = self.expmap2xyz_torch(torch.from_numpy(copy.deepcopy(expmap)).float())
                rotmat = self.rotmat_rep(action, subject, data_type)
                euler = self.euler_rep(action, subject, data_type)
                quat = self.quaternion_rep(action, subject, data_type)
                if positions.shape[0] != expmap.shape[0]:
                    print(f'''\
                            corrupted:
                            subject: {subject}
                            file: {f}
                            positions shape: {positions.shape}
                            expmap shape: {expmap.shape}
                            rotmat shape: {rotmat.shape}
                            quat shape: {quat.shape}
                    ''')
                    positions = positions[:min(positions.shape[0], expmap.shape[0])]

                # total_frame_num = self.obs_frame_num + self.pred_frame_num
                # section_range = positions.shape[0] // (
                #         total_frame_num * (self.skip_frame_num + 1)) if self.use_video_once is False else 1
                # for i in range(section_range):
                video_data = {
                    'xyz_pose': positions.reshape(positions.shape[0], -1, 3).tolist()[::self.skip_frame_num + 1],
                    'quaternion_pose': quat.reshape(quat.shape[0], -1, 4).tolist()[::self.skip_frame_num + 1],
                    'expmap_pose': expmap.reshape(expmap.shape[0], -1, 3).tolist()[::self.skip_frame_num + 1],
                    'rotmat_pose': rotmat.tolist()[::self.skip_frame_num + 1],
                    'euler_pose': euler.tolist()[::self.skip_frame_num + 1]
                    # ,'image_path': list()
                }
                # print('video',len(video_data['xyz_pose']), len(positions.tolist()), len(video_data['expmap_pose']), len(expmap.tolist()),
                # len(video_data['euler_pose']), len(euler.tolist()))

                print(f'shape {subject} {action}', positions.reshape(positions.shape[0], -1, 3).shape,
                      expmap.reshape(expmap.shape[0], -1, 3).shape,
                      euler[0].shape, rotmat[0].shape, quat.reshape(quat.shape[0], -1, 4).shape)
                # for j in range(0, positions.shape[0], self.skip_frame_num + 1):
                #     video_data['image_path'].append(f'{os.path.basename(f).split(".cdf")[0]}_{j:05}')
                
                self.update_meta_data(self.meta_data, video_data['xyz_pose'], 3)
                # print(,positions.shape,
                with jsonlines.open(os.path.join(self.output_dir, output_file_name), mode='a') as writer:
                    writer.write({
                        'video_section': f'{subject}-{canonical_name}',
                        'action': f'{canonical_name}',
                        'xyz_pose': video_data['xyz_pose'],
                        'quaternion_pose': video_data['quaternion_pose'],
                        'expmap_pose': video_data['expmap_pose'],
                        'rotmat_pose': video_data['rotmat_pose'],
                        'euler_pose': video_data['euler_pose']
                        ,'image_path': video_data['image_path']
                    })
        self.save_meta_data(self.meta_data, self.output_dir, True, data_type)
        # self.delete_redundant_files()

    def expmap_rep(self, action, subject, data_type):
        output_directory = os.path.join(PREPROCESSED_DATA_DIR, 'H3.6m_rotations')
        os.makedirs(output_directory, exist_ok=True)
        h36m_rotations_dataset_url = 'http://www.cs.stanford.edu/people/ashesh/h3.6m.zip'
        h36m_path = os.path.join(output_directory, 'h3.6m')

        if not os.path.exists(h36m_path):
            zip_path = h36m_path + ".zip"

            logger.info('Downloading Human3.6M dataset (it may take a while)...')
            if not os.path.exists(zip_path):
                urlretrieve(h36m_rotations_dataset_url, zip_path)
            if not os.path.exists(os.path.join(h36m_path, 'dataset')):
                logger.info('Extracting Human3.6M dataset...')
                with zipfile.ZipFile(zip_path, 'r') as archive:
                    archive.extractall(output_directory)
        data = self.__read_file(action, h36m_path, subject, data_type)
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
        # return data.reshape(data.shape[0], -1, 3)[:, 1:]

    @staticmethod
    def delete_redundant_files():
        output_directory = os.path.join(PREPROCESSED_DATA_DIR, 'H3.6m_rotations')
        h36_folder = os.path.join(output_directory, 'h3.6m')
        os.remove(h36_folder + ".zip")
        # rmtree(h36_folder)
