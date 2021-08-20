import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import MultipleLocator

from visualization.color_generator import color_generator
from visualization.keypoints_connection import keypoint_connections


class Visualizer:
    def __init__(self, dataset, num_keypoints):
        self.dataset = dataset
        self.num_keypoints = num_keypoints

    def visualizer_3D(self, poses: list, images_paths=None, fig_size=(8, 6)):
        """
            visualizer_3D(poses, images_paths, fig_size) -> None
            .   @brief Draws a 3D figure with matplotlib (it can have multiple sub figures).
            .
            .   The function cv::This function draw multiple 3D poses alongside each other to make comparisons.
            .   @param poses should be a list of tensors of multiple images that you want to compare.
            .   For Ex: poses.shape = [2, 5, 51] means you want to compare 2 different groups of images each contains
            .   5 person with 51 joints (17 * 3).
            .   @param images_paths list of paths to specified images (scenes).
            .   @param fig_size size of matplotlib figure.
        """

        comparison_number = len(poses)
        fig = plt.figure(figsize=fig_size, dpi=100)
        axarr = []
        for i in range(comparison_number):
            axarr.append(fig.add_subplot(1, comparison_number, i + 1, projection='3d'))
            axarr[i].xaxis.set_major_locator(MultipleLocator(0.5))
            axarr[i].yaxis.set_major_locator(MultipleLocator(0.5))
            axarr[i].zaxis.set_major_locator(MultipleLocator(0.5))
            self.__generate_3D_figure(
                all_poses=poses[i] if i < len(poses) else None,
                image_path=images_paths[i] if i < len(images_paths) else None,
                ax=axarr[i]
            )
        plt.show()

    def visualizer_2D(self, poses, masks=None, images_paths=None, fig_size=(8, 6)):
        """
             visualizer_2D(poses, masks, images_paths, fig_size) -> None
            .   @brief Draws a 2D figure with matplotlib (it can have multiple sub figures).
            .
            .   The function cv:: This function draw multiple 2D poses alongside each other to make comparisons
            .   (in different images) you can draw many different persons together in one sub figure (scene) .
            .   @param poses should be a list of tensors of multiple images that you want to compare.
            .   For Ex: poses.shape = [2, 5, 34] means you want to compare 2 different groups of images each contains
            .   5 person with 34 joints (17 * 2).
            .   @param masks: Also a list of tensors for multiple images.
            .   For Ex: masks.shape = [2,5, 17] just like 'poses'. The only difference here: we have 1 mask for each joint
            .   @param images_paths list of paths to specified images (scenes).
            .   @param fig_size size of matplotlib figure.
        """
        if masks is None:
            masks = []
        comparison_number = len(poses)
        fig = plt.figure(figsize=fig_size, dpi=100)
        images = []
        axarr = []
        for image_index in range(comparison_number):
            images.append(
                self.__generate_2D_figure(
                    all_poses=poses[image_index] if image_index < len(poses) else None,
                    all_masks=masks[image_index] if image_index < len(masks) else None,
                    image_path=images_paths[image_index] if image_index < len(images_paths) else None
                )
            )
        for i in range(comparison_number):
            axarr.append(fig.add_subplot(1, comparison_number, i + 1))
            axarr[i].imshow(images[i])
        plt.show()

    def __generate_3D_figure(self, all_poses, ax, image_path=None):
        poses = all_poses.reshape(all_poses.shape[0], self.num_keypoints, 3)
        for i, keypoints in enumerate(poses):
            for ie, edge in enumerate(keypoint_connections[self.dataset]):
                ax.plot([keypoints[edge, 0][0], keypoints[edge, 0][1]],
                        [keypoints[edge, 1][0], keypoints[edge, 1][1]],
                        [keypoints[edge, 2][0], keypoints[edge, 2][1]], linewidth=1)
            ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2])

    def __generate_2D_figure(self, all_poses, all_masks=None, image_path=None):
        if image_path is None:
            image = np.zeros((720, 1280, 3))
        else:
            image = cv2.imread(image_path)
        if all_masks is None:
            all_masks = torch.ones_like(all_poses)
        poses = all_poses.reshape(all_poses.shape[0], self.num_keypoints, 2)
        for i, keypoints in enumerate(poses):
            for keypoint in range(keypoints.shape[0]):
                if all_masks[i][keypoint // 2] != 0:
                    cv2.circle(image, (int(keypoints[keypoint, 0]), int(keypoints[keypoint, 1])), 3,
                               color_generator.get_color(2000), thickness=-1,
                               lineType=cv2.FILLED)
                    cv2.putText(image, f"{keypoint}",
                                (int(keypoints[keypoint, 0] + 10), int(keypoints[keypoint, 1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 120, 0), 1)
            for ie, edge in enumerate(keypoint_connections[self.dataset]):
                if not ((keypoints[edge, 0][0] <= 0 or keypoints[edge, 1][0] <= 0) or (
                        keypoints[edge, 0][1] <= 0 or keypoints[edge, 1][1] <= 0)) and (
                        all_masks[i][edge[0]] != 0) and (
                        all_masks[i][edge[1]] != 0):
                    cv2.line(image, (int(keypoints[edge, 0][0]), int(keypoints[edge, 1][0])),
                             (int(keypoints[edge, 0][1]), int(keypoints[edge, 1][1])),
                             color_generator.get_color(ie), 2, lineType=cv2.LINE_AA)
        return image
