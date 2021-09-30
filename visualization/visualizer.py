import math
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from matplotlib.pyplot import AutoLocator
from pygifsicle import optimize

from path_definition import ROOT_DIR
from utils.save_load import setup_visualization_dir
from visualization.color_generator import color_generator
from visualization.keypoints_connection import keypoint_connections

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, dataset_name, images_dir):
        self.images_dir = images_dir
        self.dataset_name = dataset_name

    def visualizer_3D(self, names, poses, cam_ext, cam_int, images_paths, gif_name, fig_size=(16, 12)):
        """
            visualizer_3D(poses, images_paths, fig_size) -> None
                @brief Draws a 3D figure with matplotlib (it can have multiple sub figures).
                    The function cv::This function draw multiple 3D poses alongside each other to make comparisons.
                :param names: name of each subplot. should be a list of strings.
                :param poses: torch.Tensor: should be a list of tensors of multiple outputs that you want to compare. (should have 4 dimensions)
                    shape of poses is like: [nun_comparisons, num_frames (which we create gif upon it), num_persons(in each frame), num_keypoints * dim]
                    Ex: poses.shape = [3, 16, 5, 51] means you want to compare 3 different groups of outputs each contain
                    5 persons with 17 joints (17 * 3).
                :param cam_ext: torch.Tensor or list of torch.Tensors: camera extrinsic parameters
                    shape of cam_ext is like: [num_comparisons, num_frames, 3, 4] which last two dimensions demonstrates (3, 4) matrix based on formal definitions
                    Ex: cam_ext.shape = [3, 16, 3, 4] means you want to compare 3 different groups of poses each contain
                    16 frame and unspecified number of persons. (for each frame basically we have a (3,4) matrix)
                :param cam_int: torch.Tensor or list of torch.Tensors: camera intrinsic parameters
                    shape of cam_int is like: [num_comparisons, 3, 3] which last two dimensions demonstrates (3, 3) matrix based on formal definitions
                    Ex: cam_int.shape = [3, 3] means you want to compare 3 different groups of poses each contain
                    (3, 3) matrix which demonstrate camera intrinsic parameters
                :param images_paths: list of tensors or list of numpy arrays: paths to specified outputs (scenes).
                    shape of images_paths is like: [num_comparisons, num_frames]
                    Ex: images_paths.shape = [3, 16] = means you want to compare 3 different groups of poses each have
                    16 images in it.
                :param fig_size: tuple(size=2): size of matplotlib figure.
                    Ex: (8, 6)
                :param gif_name: str: name of generated output .gif file
                :return: None: generate a .gif file
        """
        poses = self.__clean_data(poses)
        if cam_ext and cam_int is not None:
            cam_ext = self.__clean_data(cam_ext)
            if images_paths:
                images_paths = self.__generate_images_path(images_paths)
            new_pose = []
            for i, group_pose in enumerate(poses):
                new_group_pose = []
                for j in range(len(group_pose)):
                    new_group_pose.append(
                        self.__scene_to_image(group_pose[j].unsqueeze(0), cam_ext[i], cam_int).tolist())
                new_pose.append(torch.tensor(new_group_pose).squeeze(1))
            self.visualizer_2D(names=names, poses=new_pose, masks=[], images_paths=images_paths, fig_size=fig_size,
                               gif_name=gif_name + '_2D_overlay')
        logger.info("start 3D visualizing ...")
        max_axes = []
        min_axes = []
        limit_axes = []
        for i in range(3):
            max_axes.append(int(max(map(lambda sub_fig_pose: torch.max(sub_fig_pose[:, :, i::3]), poses))))
            min_axes.append(int(min(map(lambda sub_fig_pose: torch.min(sub_fig_pose[:, :, i::3]), poses))))
        for i in range(3):
            limit_axes.append([-1 * math.fabs(min_axes[i]) - 1, math.fabs(max_axes[i]) + 1])
        comparison_number = len(poses)
        axarr = []
        filenames = []
        save_dir = setup_visualization_dir(ROOT_DIR)
        for j in range(len(poses[0])):
            fig = plt.figure(figsize=fig_size, dpi=100)
            axarr.append([])
            for i in range(len(poses)):
                axarr[j].append(fig.add_subplot(1, comparison_number, i + 1, projection='3d'))
                self.__create_plot(axarr[j][i], limit_axes)
                self.__generate_3D_figure(
                    all_poses=poses[i][j],
                    ax=axarr[j][i]
                )
                for _ in range(4):
                    filenames.append(os.path.join(save_dir, f'{j}.png'))
                if j == len(poses[0]) - 1:
                    for _ in range(5):
                        filenames.append(os.path.join(save_dir, f'{j}.png'))
                plt.title(names[i])
                plt.savefig(os.path.join(save_dir, f'{j}.png'), dpi=100)
            plt.close(fig)
        with imageio.get_writer(os.path.join(save_dir, f'{gif_name}.gif'), mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(filenames):
            os.remove(filename)
        optimize(os.path.join(save_dir, f'{gif_name}.gif'))
        logger.info("end 3D visualizing ...")

    def visualizer_2D(self, names, poses, masks, images_paths, gif_name, fig_size=(8, 6)):
        """
             visualizer_2D(poses, masks, images_paths, fig_size) -> gif
                @brief Draws a 2D figure with matplotlib (it can have multiple sub figures).
                    The function cv:: This function draw multiple 2D poses alongside each other to make comparisons
                    (in different outputs) you can draw many different persons together in one sub figure (scene) .
                :param names: name of each subplot. should be a list of strings.
                :param poses: torch.Tensor or list of torch.Tensors: 3D input pose
                    shape of poses is like: [nun_comparisons, num_frames (which we create gif upon it), num_persons(in each frame), num_keypoints * dim]
                    Ex: poses.shape = [3, 16, 5, 34] means you want to compare 3 different groups of outputs each contain
                    5 persons with 17 joints (17 * 2).
                :param masks: torch.Tensor or list of torch.Tensors
                    shape of masks is like: [nun_comparisons, num_frames (which we create gif upon it), num_persons(in each frame), num_keypoints]
                    Ex: masks.shape = [3, 16, 5, 17] just like 'poses'. The only difference here: we have 1 mask for each joint
                :param images_paths: list or numpy.array: paths to specified outputs (scenes).
                    Ex: images_paths.shape = [3, 16]
                :param fig_size: tuple(size=2): size of matplotlib figure.
                    Ex: (8.6)
                :param gif_name: str: name of generated output .gif file
                :return None: generate a .gif file
        """
        logger.info("start 2D visualizing ...")
        poses = self.__clean_data(poses)
        if masks is None or masks == []:
            masks = []
        else:
            masks = self.__clean_data(masks)
        if images_paths is None or images_paths == []:
            images_paths = []
        else:
            images_paths = self.__generate_images_path(images_paths)
        subfig_size = len(poses)
        images = []
        for i, pose_group in enumerate(poses):
            images.append([])
            for j, pose in enumerate(pose_group):
                images[i].append(
                    self.__generate_2D_figure(
                        all_poses=pose,
                        all_masks=masks[i][j] if i < len(masks) and j < len(masks[i]) else None,
                        image_path=images_paths[i][j] if i < len(images_paths) and j < len(images_paths[i]) else None
                    )
                )
        filenames = []
        save_dir = setup_visualization_dir(ROOT_DIR)
        for plt_index in range(len(poses[0])):
            fig = plt.figure(figsize=fig_size, dpi=100)
            axarr = []
            for i in range(len(poses)):
                axarr.append(fig.add_subplot(1, subfig_size, i + 1))
                plt.title(names[i])
                axarr[i].imshow(images[i][plt_index])
            for i in range(4):
                filenames.append(os.path.join(save_dir, f'{plt_index}.png'))
            if plt_index == len(poses[0]) - 1:
                for i in range(5):
                    filenames.append(os.path.join(save_dir, f'{plt_index}.png'))
            plt.savefig(os.path.join(save_dir, f'{plt_index}.png'), dpi=100)
            plt.close(fig)
        with imageio.get_writer(os.path.join(save_dir, f'{gif_name}.gif'), mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in set(filenames):
            os.remove(filename)
        optimize(os.path.join(save_dir, f'{gif_name}.gif'))
        logger.info("end 2D visualizing ...")

    def __generate_3D_figure(self, all_poses, ax):
        num_keypoints = all_poses.shape[-1] // 3
        poses = all_poses.reshape(all_poses.shape[0], num_keypoints, 3)
        for i, keypoints in enumerate(poses):
            for ie, edge in enumerate(keypoint_connections[self.dataset_name]):
                pass
                ax.plot(xs=[keypoints[edge, 0][0], keypoints[edge, 0][1]],
                        zs=[keypoints[edge, 1][0], keypoints[edge, 1][1]],
                        ys=[keypoints[edge, 2][0], keypoints[edge, 2][1]], linewidth=1, label=r'$z=y=x$')
            ax.scatter(xs=keypoints[:, 0].detach().cpu().numpy(), zs=keypoints[:, 1].detach().cpu().numpy(),
                       ys=keypoints[:, 2].detach().cpu().numpy(), s=1)

    def __generate_2D_figure(self, all_poses, all_masks=None, image_path=None):
        num_keypoints = all_poses.shape[-1] // 2
        poses = all_poses.reshape(all_poses.shape[0], num_keypoints, 2)
        if image_path is None:
            image = np.zeros((1080, 1920, 3)).astype(np.uint8)
        else:
            image = cv2.imread(image_path)
        if all_masks is None:
            all_masks = torch.ones(all_poses.shape[0], all_poses.shape[1] // 2)
        for i, keypoints in enumerate(poses):
            for keypoint in range(keypoints.shape[0]):
                if all_masks[i][keypoint // 2] != 0:
                    cv2.circle(image, (int(keypoints[keypoint, 0]), int(keypoints[keypoint, 1])), 3,
                               (0, 255, 255), thickness=-1,
                               lineType=cv2.FILLED)
                    cv2.putText(image, f"{keypoint}",
                                (int(keypoints[keypoint, 0] + 10), int(keypoints[keypoint, 1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            for ie, edge in enumerate(keypoint_connections[self.dataset_name]):
                if not ((keypoints[edge, 0][0] <= 0 or keypoints[edge, 1][0] <= 0) or (
                        keypoints[edge, 0][1] <= 0 or keypoints[edge, 1][1] <= 0)) and (
                        all_masks[i][edge[0]] != 0) and (
                        all_masks[i][edge[1]] != 0):
                    cv2.line(image, (int(keypoints[edge, 0][0]), int(keypoints[edge, 1][0])),
                             (int(keypoints[edge, 0][1]), int(keypoints[edge, 1][1])),
                             color_generator.get_color(ie), 4, lineType=cv2.LINE_AA)
        return image

    @staticmethod
    def __create_plot(axe, axes_limit):
        axe.xaxis.set_major_locator(AutoLocator())
        axe.yaxis.set_major_locator(AutoLocator())
        axe.zaxis.set_major_locator(AutoLocator())
        axe.set_aspect('auto')
        max_range = np.array([max(axes_limit[i]) - min(axes_limit[i]) for i in range(len(axes_limit))]).max() / 2.0
        x_mean = (max(axes_limit[0]) + min(axes_limit[0])) * 0.5
        y_mean = (max(axes_limit[1]) + min(axes_limit[1])) * 0.5
        z_mean = (max(axes_limit[2]) + min(axes_limit[2])) * 0.5
        axe.set_xlim(xmin=x_mean - max_range, xmax=x_mean + max_range)
        axe.set_ylim(ymin=y_mean - max_range, ymax=y_mean + max_range)
        axe.set_zlim(zmin=z_mean - max_range, zmax=z_mean + max_range)

    @staticmethod
    def __scene_to_image(pose, cam_ext, cam_int):
        """
            scene_to_image(pose, cam_ext, cam_int) -> 2D_pose
                @brief this function project 3D locations with respect to camera into 2D pixels on body poses.
                :param pose: torch.Tensor: 3D input pose
                    shape of pose is like: [num_persons(in each frame), num_frames, num_keypoints * 3]
                    Ex: [2, 16, 72]
                :param cam_ext: torch.Tensor: camera extrinsic parameters
                    shape of cam_ext is like: [num_frames, 3, 4] which last two dimensions demonstrates (3, 4) matrix based on formal definitions
                    Ex: [16, 3, 4]
                :param cam_int: torch.Tensor: camera intrinsic parameters
                    shape of cam_int is like: [3, 3] which demonstrate (3, 3) matrix based on formal definitions
                :return 2d_pose: torch.Tensor: 2D projected pose
        """
        first_shape = pose.shape
        poses = pose.reshape(pose.shape[0], pose.shape[1], pose.shape[-1] // 3, 3)
        one_padding = torch.ones(poses.shape[0], poses.shape[1], pose.shape[-1] // 3, 1)

        poses = torch.cat((poses, one_padding), 3)
        poses = poses.transpose(1, 0)
        new_pose = []
        for frame_num, frame_data in enumerate(poses):
            for p_data in frame_data:
                new_data = []
                for joint_data in p_data:
                    new_joint_data = torch.matmul(cam_int, torch.matmul(cam_ext[frame_num][:3], joint_data))
                    new_data.append((new_joint_data[:2] / new_joint_data[-1]).tolist())
                new_pose.append(new_data)
        return torch.tensor(new_pose).reshape(first_shape[0], first_shape[1], 2 * first_shape[-1] // 3)

    @staticmethod
    def __clean_data(input_data: list):
        new_data = []
        max_len = len(input_data[0])

        for i in range(len(input_data)):
            if len(input_data[i]) > max_len:
                max_len = len(input_data[i])
        for i, pose in enumerate(input_data):
            if len(input_data[i]) < max_len:
                size = [1 for _ in range(len(pose.shape))]
                size[0] = max_len - len(input_data[i])
                last_row = pose[-1:]
                expended_data = last_row.repeat(size)
                expanded_data = torch.cat((pose, expended_data))
            else:
                expanded_data = pose
            new_data.append(expanded_data)
        return new_data

    def __generate_images_path(self, images_paths):
        new_images_path = []
        max_len = len(images_paths[0])
        for i in range(len(images_paths)):
            if len(images_paths[i]) > max_len:
                max_len = len(images_paths[i])
        for i, image_path in enumerate(images_paths):
            group_images_path = []
            for img in image_path:
                group_images_path.append(os.path.join(self.images_dir, img))
            if len(image_path) < max_len:
                last_path = image_path[-1]
                for i in range(max_len - len(image_path)):
                    group_images_path.append(os.path.join(self.images_dir, last_path))
            new_images_path.append(group_images_path)
        return new_images_path
