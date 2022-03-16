import json
import os
from json import JSONEncoder

import numpy as np


class Processor:
    def __init__(self, dataset_path, is_interactive, obs_frame_num, pred_frame_num, skip_frame_num,
                 use_video_once, custom_name, save_total_frames):
        self.is_interactive = is_interactive
        self.obs_frame_num = obs_frame_num
        self.pred_frame_num = pred_frame_num
        self.skip_frame_num = skip_frame_num
        self.use_video_once = use_video_once
        self.dataset_path = dataset_path
        self.custom_name = custom_name
        self.save_total_frames = save_total_frames

    def normal(self, data_type='train'):
        """
        :param data_type: specify what kind of static file you want to creat (options are: <train>, <test>, <validation>
        :return: None: create static <.csv> file
        """
        pass

    @staticmethod
    def update_meta_data(meta_data, new_data, dim):
        """
        :param meta_data: pass existing meta_data
        :param new_data: pass new data
        :param dim: pass dimension of joint i.e. 2 for 2D or 3 for 3D
        :return None: update meta_data
        """
        np_data = np.array(new_data)
        if len(np_data.shape) == 2:
            np_data = np.expand_dims(np_data, axis=0)

        meta_data['avg_person'].append(np_data.shape[0])
        meta_data['count'] += np_data.size // dim
        meta_data['sum2_pose'] += np.array([np.sum(np.square(np_data[:, :, i::dim])) for i in range(dim)])
        meta_data['sum_pose'] += np.array([np.sum(np_data[:, :, i::dim]) for i in range(dim)])
        np_data = np_data.reshape(*np_data.shape[:-1], np_data.shape[-1] // dim, dim)
        new_max = [np.max(np_data[:, :, :, i]) for i in range(dim)]
        new_min = [np.min(np_data[:, :, :, i]) for i in range(dim)]
        for i in range(dim):
            if new_max[i] > meta_data['max_pose'][i]:
                meta_data['max_pose'][i] = new_max[i]
            if new_min[i] < meta_data['min_pose'][i]:
                meta_data['min_pose'][i] = new_min[i]

    @staticmethod
    def save_meta_data(meta_data, outputdir, is_3d, data_type):
        """
        :param meta_data: pass existing and also final meta data
        :param outputdir: pass output directory in which you want to save meta data
        :param is_3d: pass if your data is in 3D format or not (3D or 2D)
        :return None: save meta data as json format
        """
        if data_type != 'train':
            return
        assert meta_data['count'] > 0
        output_file_path = os.path.join(outputdir, f'3D_meta.json' if is_3d else f'2D_meta.json')
        meta = {
            'avg_person': np.mean(np.array(meta_data['avg_person'])),
            'std_person': np.std(np.array(meta_data['avg_person'])),
            'avg_pose': meta_data['sum_pose'] / meta_data['count'],
            'max_pose': meta_data['max_pose'],
            'min_pose': meta_data['min_pose'],
            'std_pose': np.sqrt(
                ((meta_data['sum2_pose'] - (np.square(meta_data['sum_pose']) / meta_data['count'])) /
                 meta_data['count']))
        }
        with open(output_file_path, 'w') as f_object:
            json.dump(meta, f_object, cls=NumpyEncoder, indent=4)


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
