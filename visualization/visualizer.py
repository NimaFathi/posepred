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


# if __name__ == '__main__':
#     vis = Visualizer('JTA', 22)
#     x = torch.tensor([
#         [31.5, 136.0999984741211, 28.5, 143.0999984741211, 38.5, 126.0999984741211, 0, 0, 0, 0,
#          49.59474921366457, 131.43147222481318, 0, 0, 68.63686040004097, 171.52201127552243, 0, 0,
#          70.63686040004097, 199.57465025849294, 0, 0, 22.54421579001289, 190.5599113432612, 0, 0, 0, 0, 0,
#          0, 42.57369362047637, 370.24436185650086, 0, 0],
#         [294.5, 158.0999984741211, 279.5, 165.0999984741211, 302.5, 144.5999984741211, 0, 0, 0, 0, 296.5,
#          170.0999984741211, 255.5, 148.0999984741211, 301.5, 191.0999984741211, 229.5, 183.0999984741211, 308.5,
#          225.0999984741211, 0, 0, 267.5, 220.0999984741211, 241.5, 216.0999984741211, 286.5, 270.10000014305115, 245.5,
#          264.10000014305115, 253.5, 317.10000014305115, 231.5, 335.10000014305115]
#     ])
#     xx = torch.tensor([[782., 213., 785., 221., 788., 229., 784., 231., 775., 232., 770.,
#                         249., 765., 268., 789., 232., 801., 235., 810., 254., 809., 272.,
#                         790., 247., 790., 256., 789., 262., 789., 268., 789., 270., 783.,
#                         275., 774., 304., 792., 317., 794., 276., 794., 306., 800., 335.],
#                        [782., 213., 785., 221., 788., 229., 784., 231., 775., 232., 770.,
#                         249., 765., 268., 789., 232., 801., 235., 810., 254., 809., 272.,
#                         790., 247., 790., 256., 789., 262., 789., 268., 789., 270., 783.,
#                         275., 774., 304., 792., 317., 794., 276., 794., 306., 800., 335.]])
#     xxx = torch.tensor([[-2.37, -4.34, 15.4, -2.33, -4.26, 15.47, -2.31, -4.18, 15.55,
#                          -2.36, -4.14, 15.55, -2.5, -4.17, 15.66, -2.6, -3.98, 15.84,
#                          -2.67, -3.74, 15.9, -2.3, -4.14, 15.52, -2.13, -4.09, 15.49,
#                          -2.01, -3.84, 15.52, -2.01, -3.58, 15.5, -2.3, -3.96, 15.67,
#                          -2.31, -3.86, 15.71, -2.32, -3.78, 15.73, -2.32, -3.69, 15.75,
#                          -2.33, -3.68, 15.76, -2.42, -3.62, 15.82, -2.56, -3.25, 15.91,
#                          -2.36, -3.13, 16.27, -2.25, -3.59, 15.73, -2.27, -3.2, 15.83,
#                          -2.22, -2.85, 16.05],
#                         [-4.11, -3.24, 12.67, -4.07, -3.15, 12.73, -4.03, -3.07, 12.8,
#                          -4.07, -3.03, 12.79, -4.21, -3.03, 12.91, -4.29, -2.82, 13.07,
#                          -4.38, -2.58, 13.12, -4.01, -3.02, 12.76, -3.83, -2.99, 12.73,
#                          -3.7, -2.76, 12.77, -3.71, -2.5, 12.76, -4., -2.85, 12.9,
#                          -3.99, -2.74, 12.94, -4.01, -2.66, 12.96, -4.01, -2.58, 12.99,
#                          -4.01, -2.56, 12.99, -4.11, -2.51, 13.04, -4.17, -2.11, 13.08,
#                          -4.13, -1.74, 13.26, -3.94, -2.47, 12.98, -3.86, -2.08, 13.08,
#                          -3.7, -2.01, 13.46]
#                         ])
#     y = []
#     all_masks = []
#     y.append(xxx)
#     masks = torch.randint(low=0, high=2, size=(2, 17))
#     all_masks.append(masks)
#     all_masks.append(masks)
#     vis.visualizer_3D(y, images_paths=[])
