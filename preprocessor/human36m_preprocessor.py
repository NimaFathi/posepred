import csv
import logging
import os
import zipfile
from glob import glob
from urllib.request import urlretrieve

import cdflib
import jsonlines
import numpy as np

from path_definition import PREPROCESSED_DATA_DIR
from preprocessor.preprocessor import Processor
from utils.others import expmap_to_quaternion, qfix, expmap_to_rotmat, expmap_to_euler

logger = logging.getLogger(__name__)

SPLIT = {
    'train': ['S1', 'S5', 'S6', 'S7', 'S8'],
    'validation': ['S1', 'S5', 'S6', 'S7', 'S8'],
    'test': ['S9', 'S11']
}


class PreprocessorHuman36m(Processor):
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name):
        super(PreprocessorHuman36m, self).__init__(dataset_path, is_interactive, obs_frame_num,
                                                   pred_frame_num, skip_frame_num, use_video_once, custom_name)
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

    def normal(self, data_type='train'):
        self.subjects = SPLIT[data_type]
        logger.info(
                'start creating Human3.6m normal static data \
                        from original Human3.6m dataset (CDF files) ... ')
        if self.custom_name:
            output_file_name = \
                    f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_{self.custom_name}.jsonl'
        else:
            output_file_name = \
                    f'{data_type}_{self.obs_frame_num}_{self.pred_frame_num}_{self.skip_frame_num}_human3.6m.jsonl'
        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"
        for subject in self.subjects:
            logger.info("handling subject: {}".format(subject))
            subject_pose_path = os.path.join(self.dataset_path, subject, 'MyPoseFeatures/D3_Positions/*.cdf')
            print('subject_pose_path:', subject_pose_path)
            file_list_pose = glob(subject_pose_path)
            assert len(file_list_pose) == 30, "Expected 30 files for subject " + subject + ", got " + str(
                len(file_list_pose))
            for f in file_list_pose:
                action = os.path.splitext(os.path.basename(f))[0]
                print(action)
                if subject == 'S11' and action == 'Directions':
                    continue  # Discard corrupted video
                canonical_name = action.replace('TakingPhoto', 'Photo') \
                    .replace('WalkingDog', 'WalkDog')
                hf = cdflib.CDF(f)
                positions = hf['Pose'].reshape(-1, 96)
                if data_type == 'train':
                    positions = positions[:95 * positions.shape[0] // 100]
                elif data_type == 'validation':
                    positions = positions[95 * positions.shape[0] // 100:]
                positions /= 1000
                expmap = self.expmap_rep(f, subject, data_type)
                rotmat = self.rotmat_rep(f, subject, data_type)
                euler = self.euler_rep(f, subject, data_type)
                #print('rotmat shape', rotmat.shape)
                #print(expmap.shape)
                quat = self.quaternion_rep(f, subject, data_type)
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
                
                total_frame_num = self.obs_frame_num + self.pred_frame_num
                section_range = positions.shape[0] // (
                        total_frame_num * (self.skip_frame_num + 1)) if self.use_video_once is False else 1
                for i in range(section_range):
                    video_data = {
                        'observed_pose': list(),
                        'future_pose': list(),
                        'observed_quaternion_pose': list(),
                        'future_quaternion_pose': list(),
                        'observed_expmap_pose': list(),
                        'future_expmap_pose': list(),
                        'observed_rotmat_pose': list(),
                        'future_rotmat_pose': list(),
                        'observed_euler_pose': list(),
                        'future_euler_pose': list(),
                        'observed_image_path': list(),
                        'future_image_path': list()
                    }
                    for j in range(0, total_frame_num * (self.skip_frame_num + 1), self.skip_frame_num + 1):
                        if j < (self.skip_frame_num + 1) * self.obs_frame_num:
                            video_data['observed_pose'].append(
                                positions[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                            # video_data['observed_quaternion_pose'].append(
                            #     quat[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                            video_data['observed_expmap_pose'].append(
                                expmap[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                            video_data['observed_rotmat_pose'].append(
                                rotmat[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                            video_data['observed_euler_pose'].append(
                                euler[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())                                         
                            video_data['observed_image_path'].append(
                                f'{os.path.basename(f).split(".cdf")[0]}_{i * total_frame_num * (self.skip_frame_num + 1) + j:05}')
                        else:
                            video_data['future_pose'].append(
                                positions[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                            # video_data['future_quaternion_pose'].append(
                            #     quat[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                            video_data['future_expmap_pose'].append(
                                expmap[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                            video_data['future_rotmat_pose'].append(
                                rotmat[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                            video_data['future_euler_pose'].append(
                                euler[i * total_frame_num * (self.skip_frame_num + 1) + j].tolist())
                            video_data['future_image_path'].append(
                                f'{os.path.basename(f).split(".cdf")[0]}_{i * total_frame_num * (self.skip_frame_num + 1) + j:05}'
                            )
                    self.update_meta_data(self.meta_data, video_data['observed_pose'], 3)
                    with jsonlines.open(os.path.join(self.output_dir, output_file_name), mode='a') as writer:
                        writer.write({
                            'video_section': f'{subject}-{canonical_name}-{i}',
                            'observed_pose': video_data['observed_pose'],
                            'future_pose': video_data['future_pose'],
                            # 'observed_quaternion_pose': video_data['observed_quaternion_pose'],
                            # 'future_quaternion_pose': video_data['future_quaternion_pose'],
                            'observed_expmap_pose': video_data['observed_expmap_pose'],
                            'future_expmap_pose': video_data['future_expmap_pose'],
                            'observed_rotmat_pose': video_data['observed_rotmat_pose'],
                            'future_rotmat_pose': video_data['future_rotmat_pose'],
                            'observed_euler_pose': video_data['observed_euler_pose'],
                            'future_euler_pose': video_data['future_euler_pose'],                            
                            'observed_image_path': video_data['observed_image_path'],
                            'future_image_path': video_data['future_image_path'],
                            'action': action.split()[0]
                        })
        self.save_meta_data(self.meta_data, self.output_dir, True, data_type)
        # self.delete_redundant_files()
    
    def expmap_rep(self, file_path, subject, data_type):
        output_directory = os.path.join(PREPROCESSED_DATA_DIR, 'H3.6m_rotations')
        os.makedirs(output_directory, exist_ok=True)
        h36m_rotations_dataset_url = 'http://www.cs.stanford.edu/people/ashesh/h3.6m.zip'
        h36m_path = os.path.join(output_directory, 'h3.6m')
        #print('h36m_path:', h36m_path)
        if not os.path.exists(h36m_path):
            zip_path = h36m_path + ".zip"

            logger.info('Downloading Human3.6M dataset (it may take a while)...')
            if not os.path.exists(zip_path):
                urlretrieve(h36m_rotations_dataset_url, zip_path)
            if not os.path.exists(os.path.join(h36m_path, 'dataset')):
                logger.info('Extracting Human3.6M dataset...')
                with zipfile.ZipFile(zip_path, 'r') as archive:
                    archive.extractall(output_directory)
        data = self.__read_file(file_path, h36m_path, subject, data_type)
        return data
    
    def rotmat_rep(self, file_path, subject, data_type):
        data = self.expmap_rep(file_path, subject, data_type)
        data = expmap_to_rotmat(data)
        return data
    
    def euler_rep(self, file_path, subject, data_type):
        data = self.expmap_rep(file_path, subject, data_type)
        data = expmap_to_euler(data)
        return data

    def quaternion_rep(self, file_path, subject, data_type):
        data = self.expmap_rep(file_path, subject, data_type)
        quat = expmap_to_quaternion(-data)
        quat = qfix(quat)
        quat = quat.reshape(-1, 32 * 4)
        return quat.reshape(-1, 32 * 4)

    @staticmethod
    def __read_file(file_path, rot_dir_path, subject, data_type):
        '''
        Read an individual file in expmap format,
        and return a NumPy tensor with shape (sequence length, number of joints, 3).
        '''
        action = os.path.splitext(os.path.basename(file_path))[0].replace('WalkTogether', 'WalkingTogether').replace('WalkDog', 'WalkingDog')
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
        return data.reshape(data.shape[0], -1, 3)[:, 1:]

    @staticmethod
    def delete_redundant_files():
        output_directory = os.path.join(PREPROCESSED_DATA_DIR, 'H3.6m_rotations')
        h36_folder = os.path.join(output_directory, 'h3.6m')
        os.remove(h36_folder + ".zip")
        # rmtree(h36_folder)