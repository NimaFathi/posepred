import os
import json
import numpy as np
from args.helper import NumpyEncoder
from path_definition import ROOT_DIR
OUTPUT_DIR = os.path.join(ROOT_DIR, 'preprocessed_data/')


class Processor:
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name):
        self.is_interactive = is_interactive
        self.obs_frame_num = obs_frame_num
        self.pred_frame_num = pred_frame_num
        self.skip_frame_num = skip_frame_num
        self.use_video_once = use_video_once
        self.dataset_path = dataset_path
        self.custom_name = custom_name



    def normal(self, data_type='train'):
        """
        :param data_type: specify what kind of static file you want to creat (options are: <train>, <test>, <validation>
        :return: None: create static <.csv> file
        """
        pass

    @staticmethod
    def update_meta_data(meta_data, new_data, dim):
        np_data = np.array(new_data)
        meta_data['avg_person'].append(np_data.shape[0])
        meta_data['var_pose'].append([np.var(np_data[:, :, i::dim]) for i in range(dim)])
        meta_data['mean_pose'].append([np.mean(np_data[:, :, i::dim]) for i in range(dim)])

    @staticmethod
    def save_meta_data(meta_data, outputdir, is_3d):
        output_file_path = os.path.join(outputdir, f'3D_meta.txt' if is_3d else f'2D_meta.txt')
        meta = {
            'avg_person': np.mean(np.array(meta_data['avg_person'])),
            'std_person': np.std(np.array(meta_data['avg_person'])),
            'avg_pose': meta_data['sum_pose'] / meta_data['count'],
            'std_pose': np.sqrt(((meta_data['sum2_pose'] - (meta_data['sum_pose'] / meta_data['count'])) / meta_data['count']))
        }
        with open(output_file_path, 'w') as f_object:
            json.dump(meta, f_object, cls=NumpyEncoder, indent=4)

