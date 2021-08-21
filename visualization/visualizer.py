import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import MultipleLocator
from pygifsicle import optimize
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
            .   @param poses should be a list of tensors of multiple outputs that you want to compare. (should have 4 dimensions)
            .   For Ex: poses.shape = [3, 16, 3, 24] means you want to compare 3 different groups of outputs each contains
            .   3 person with 24 joints (17 * 3).
            .   @param images_paths list of paths to specified outputs (scenes).
            .   @param fig_size size of matplotlib figure.
        """

        comparison_number = len(poses)
        fig = plt.figure(figsize=fig_size, dpi=200)
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

    def visualizer_2D(self, poses=None, masks=None, images_paths=None, fig_size=(8, 6), name='2D_visualize'):
        """
             visualizer_2D(poses, masks, images_paths, fig_size) -> gif
            .   @brief Draws a 2D figure with matplotlib (it can have multiple sub figures).
            .
            .   The function cv:: This function draw multiple 2D poses alongside each other to make comparisons
            .   (in different outputs) you can draw many different persons together in one sub figure (scene) .
            .   @param poses should be a list of tensors of multiple outputs that you want to compare.
            .   shape of poses is like: [nun_comparisons, num_frames, num_persons(in each frame), num_keypoints * 2]
            .   For Ex: poses.shape = [3, 16, 4, 34] means you want to compare 2 different groups of outputs each contains
            .   4 person with 34 joints in 16 frames (17 * 2).
            .   @param masks: Also a list of tensors for multiple outputs.
            .   shape os masks is like: [nun_comparisons, num_frames (which we create gif upon it), num_persons(in each frame), num_keypoints]
            .   For Ex: masks.shape = [3, 16, 5, 17] just like 'poses'. The only difference here: we have 1 mask for each joint
            .   @param images_paths list of paths to specified outputs (scenes).
            .   @param fig_size size of matplotlib figure.
        """
        if masks is None:
            masks = []
        subfig_size = len(poses)
        images = []
        for i, pose_group in enumerate(poses):
            images.append([])
            for j, pose in enumerate(pose_group):
                images[i].append(
                    self.__generate_2D_figure(
                        all_poses=pose,
                        all_masks=masks[i][j] if i < len(masks) and j < len(masks[i]) else None,
                        image_path=images_paths[j] if j < len(images_paths) else None
                    )
                )
        filenames = []
        for plt_index in range(len(poses[0])):
            fig = plt.figure(figsize=fig_size, dpi=100)
            axarr = []
            for i in range(len(poses)):
                axarr.append(fig.add_subplot(1, subfig_size, i + 1))
                axarr[i].imshow(images[i][plt_index])
            for i in range(2):
                filenames.append(f'./outputs/{plt_index}.png')
            if plt_index == len(poses[0]) - 1:
                for i in range(5):
                    filenames.append(f'./outputs/{plt_index}.png')
            plt.savefig(f'./outputs/{plt_index}.png', dpi=100)
        with imageio.get_writer(f'./outputs/{name}.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        # Remove files
        for filename in set(filenames):
            os.remove(filename)
        optimize(f'./outputs/{name}.gif')

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
            image = np.zeros((720, 1080, 3))
        else:
            image = cv2.imread(image_path)
        if all_masks is None:
            all_masks = torch.ones(all_poses.shape[0], all_poses.shape[1] // 2)
        poses = all_poses.reshape(all_poses.shape[0], self.num_keypoints, 2)
        for i, keypoints in enumerate(poses):
            for keypoint in range(keypoints.shape[0]):
                if all_masks[i][keypoint // 2] != 0:
                    cv2.circle(image, (int(keypoints[keypoint, 0]), int(keypoints[keypoint, 1])), 5,
                               color_generator.get_color(2000), thickness=-1,
                               lineType=cv2.FILLED)
                    cv2.putText(image, f"{keypoint}",
                                (int(keypoints[keypoint, 0] + 10), int(keypoints[keypoint, 1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            for ie, edge in enumerate(keypoint_connections[self.dataset]):
                if not ((keypoints[edge, 0][0] <= 0 or keypoints[edge, 1][0] <= 0) or (
                        keypoints[edge, 0][1] <= 0 or keypoints[edge, 1][1] <= 0)) and (
                        all_masks[i][edge[0]] != 0) and (
                        all_masks[i][edge[1]] != 0):
                    cv2.line(image, (int(keypoints[edge, 0][0]), int(keypoints[edge, 1][0])),
                             (int(keypoints[edge, 0][1]), int(keypoints[edge, 1][1])),
                             color_generator.get_color(ie), 4, lineType=cv2.LINE_AA)
        return image


# if __name__ == '__main__':
#     vis = Visualizer('PoseTrack', 17)
#     image_paths = ['/home/nima/EPFL/012940_mpii_train/000050.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000051.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000052.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000053.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000054.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000055.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000056.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000057.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000058.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000059.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000060.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000061.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000062.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000063.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000064.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000065.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000066.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000067.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000068.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000069.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000070.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000071.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000072.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000073.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000074.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000075.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000076.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000077.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/000078.jpg',
#                    '/home/nima/EPFL/012940_mpii_train/0940_mpii_train/000079.jpg']
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
#     xxx = torch.tensor(
#         [[[31.5, 136.0999984741211, 28.5, 143.0999984741211, 38.5, 126.0999984741211, 0, 0, 0, 0, 49.59474921366457,
#            131.43147222481318, 0, 0, 68.63686040004097, 171.52201127552243, 0, 0, 70.63686040004097, 199.57465025849294,
#            0, 0, 22.54421579001289, 190.5599113432612, 0, 0, 0, 0, 0, 0, 42.57369362047637, 370.24436185650086, 0, 0],
#           [38.5, 134.0999984741211, 26.5, 139.0999984741211, 41.5, 123.0999984741211, 0, 0, 0, 0, 52.59474921366457,
#            132.43147222481318, 0, 0, 68.63686040004097, 171.52201127552243, 0, 0, 70.63686040004097, 199.57465025849294,
#            0, 0, 21.54421579001289, 190.5599113432612, 0, 0, 0, 0, 0, 0, 40.57369362047637, 372.24436185650086, 0, 0],
#           [38.5, 127.0999984741211, 26.5, 134.0999984741211, 43.5, 115.0999984741211, 0, 0, 0, 0, 53.59474921366457,
#            130.93147222481318, 0, 0, 68.63686040004097, 172.52201127552243, 0, 0, 71.63686040004097, 199.57465025849294,
#            0, 0, 21.54421579001289, 189.5599113432612, 0, 0, 0, 0, 0, 0, 39.57369362047637, 375.24436185650086, 0, 0],
#           [38.5, 125.0999984741211, 26.5, 132.0999984741211, 45.5, 111.0999984741211, 0, 0, 0, 0, 54.59474921366457,
#            131.43147222481318, 0, 0, 67.63686040004097, 173.52201127552243, 0, 0, 69.63686040004097, 199.57465025849294,
#            0, 0, 20.54421579001289, 188.5599113432612, 0, 0, 0, 0, 0, 0, 36.57369362047637, 378.24436185650086, 0, 0],
#           [41.5, 121.0999984741211, 26.5, 129.0999984741211, 44.5, 106.0999984741211, 0, 0, 0, 0, 54.59474921366457,
#            129.43147222481318, 0, 0, 67.63686040004097, 174.52201127552243, 0, 0, 68.63686040004097, 202.57465025849294,
#            0, 0, 19.54421579001289, 188.5599113432612, 0, 0, 0, 0, 0, 0, 34.57369362047637, 380.24436185650086, 0, 0],
#           [42.5, 115.0999984741211, 26.5, 125.0999984741211, 45.5, 103.0999984741211, 0, 0, 0, 0, 53.59474921366457,
#            127.43147222481318, 0, 0, 68.63686040004097, 174.52201127552243, 0, 0, 69.63686040004097, 202.57465025849294,
#            0, 0, 20.54421579001289, 186.5599113432612, 0, 0, 0, 0, 0, 0, 32.57369362047637, 383.24436185650086, 0, 0],
#           [42.5, 114.0999984741211, 26.5, 124.0999984741211, 45.5, 101.0999984741211, 0, 0, 0, 0, 54.59474921366457,
#            121.43147222481318, 0, 0, 68.63686040004097, 173.52201127552243, 0, 0, 69.63686040004097, 204.57465025849294,
#            0, 0, 21.54421579001289, 185.5599113432612, 0, 0, 0, 0, 0, 0, 34.57369362047637, 385.24436185650086, 0, 0],
#           [41.5, 112.0999984741211, 26.5, 122.0999984741211, 43.5, 96.0999984741211, 0, 0, 0, 0, 53.59474921366457,
#            118.43147222481318, 0, 0, 70.63686040004097, 173.52201127552243, 0, 0, 68.63686040004097, 206.57465025849294,
#            0, 0, 21.54421579001289, 185.5599113432612, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [39.5, 108.0999984741211, 25.5, 120.0999984741211, 41.5, 93.0999984741211, 0, 0, 0, 0, 57.59474921366457,
#            119.43147222481318, 0, 0, 70.63686040004097, 173.52201127552243, 0, 0, 70.63686040004097, 208.57465025849294,
#            0, 0, 20.54421579001289, 185.5599113432612, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [42.5, 105.0999984741211, 27.5, 120.0999984741211, 42.5, 91.0999984741211, 0, 0, 0, 0, 59.59474921366457,
#            115.43147222481318, 0, 0, 71.63686040004097, 171.52201127552243, 0, 0, 70.63686040004097, 207.57465025849294,
#            0, 0, 20.54421579001289, 183.5599113432612, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [41.5, 99.0999984741211, 30.5, 112.0999984741211, 44.5, 84.0999984741211, 0, 0, 0, 0, 60.59474921366457,
#            111.43147222481318, 0, 0, 72.63686040004097, 169.52201127552243, 0, 0, 74.63686040004097, 208.57465025849294,
#            0, 0, 23.54421579001289, 182.5599113432612, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [46.5, 96.0999984741211, 33.5, 112.0999984741211, 47.5, 81.0999984741211, 0, 0, 0, 0, 62.59474921366457,
#            108.43147222481318, 0, 0, 75.63686040004097, 167.52201127552243, 0, 0, 77.63686040004097, 207.57465025849294,
#            0, 0, 26.54421579001289, 182.5599113432612, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [49.5, 90.0999984741211, 34.5, 112.0999984741211, 52.5, 75.0999984741211, 0, 0, 0, 0, 65.59474921366457,
#            105.43147222481318, 0, 0, 80.63686040004097, 166.52201127552243, 0, 0, 84.63686040004097, 208.57465025849294,
#            0, 0, 28.544215790012885, 175.5599113432612, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [50.5, 90.0999984741211, 30.5, 106.0999984741211, 54.5, 69.0999984741211, 0, 0, 0, 0, 71.59474921366457,
#            101.43147222481318, 0, 0, 85.63686040004097, 164.52201127552243, 0, 0, 92.63686040004097, 208.57465025849294,
#            0, 0, 33.544215790012885, 173.5599113432612, 13.523160196824687, 170.5072723602907, 0, 0, 0, 0, 0, 0, 0, 0],
#           [55.5, 82.0999984741211, 35.5, 99.0999984741211, 62.5, 60.099998474121094, 0, 0, 0, 0, 76.59474921366457,
#            100.43147222481318, 0, 0, 92.63686040004097, 163.52201127552243, 0, 0, 103.63686040004097,
#            209.57465025849294, 0, 0, 37.544215790012885, 170.5599113432612, 16.523160196824687, 166.5072723602907, 0, 0,
#            0, 0, 0, 0, 0, 0],
#           [59.5, 86.0999984741211, 42.5, 92.0999984741211, 66.5, 57.099998474121094, 0, 0, 0, 0, 83.59474921366457,
#            98.43147222481318, 0, 0, 101.63686040004097, 161.52201127552243, 0, 0, 114.63686040004097,
#            209.57465025849294, 0, 0, 40.544215790012885, 168.5599113432612, 17.523160196824687, 165.5072723602907, 0, 0,
#            0, 0, 0, 0, 0, 0]], [
#              [187.89163307761822, 122.39988883503088, 176.86215524715473, 133.43568334345082, 192.89584419625584,
#               107.38093880116149, 0, 0, 0, 0, 203.92742758603816, 125.43147222481318, 142.78846067099602,
#               117.40831107230616, 218.94848317922637, 174.97330511649383, 115.7337161287067, 178.9859384724068,
#               233.98006656900867, 221.61676144486935, 132.76740507780784, 221.06173860788428, 165.5, 186.0999984741211,
#               134.5, 194.0999984741211, 198.91689978944407, 237.0190670093871, 136.77793287440193, 276.7367783260421,
#               159.82214962009715, 307.80415622424437, 123.75056060325727, 364.9220675460983],
#              [193.89163307761822, 119.39988883503088, 182.86215524715473, 129.43568334345082, 199.89584419625584,
#               104.38093880116149, 0, 0, 0, 0, 206.92742758603816, 122.43147222481318, 146.78846067099602,
#               116.40831107230616, 222.94848317922637, 170.97330511649383, 119.7337161287067, 180.9859384724068,
#               239.98006656900867, 221.61676144486935, 136.76740507780784, 219.06173860788428, 170.5, 184.0999984741211,
#               135.5, 194.0999984741211, 207.91689978944407, 238.0190670093871, 137.77793287440193, 281.7367783260421,
#               162.82214962009715, 307.80415622424437, 128.75056060325727, 365.9220675460983],
#              [196.89163307761822, 116.39988883503088, 185.86215524715473, 127.43568334345082, 201.89584419625584,
#               102.38093880116149, 0, 0, 0, 0, 211.92742758603816, 123.43147222481318, 150.78846067099602,
#               113.40831107230616, 225.94848317922637, 170.97330511649383, 121.7337161287067, 180.9859384724068,
#               243.98006656900867, 219.61676144486935, 139.76740507780784, 219.06173860788428, 170.5, 182.0999984741211,
#               137.5, 192.0999984741211, 205.91689978944407, 237.0190670093871, 140.77793287440193, 281.7367783260421,
#               169.82214962009715, 309.80415622424437, 129.75056060325727, 366.9220675460983],
#              [199.89163307761822, 112.39988883503088, 188.86215524715473, 124.43568334345082, 205.89584419625584,
#               98.38093880116149, 0, 0, 0, 0, 211.92742758603816, 120.43147222481318, 153.78846067099602,
#               112.40831107230616, 227.94848317922637, 169.97330511649383, 123.7337161287067, 178.9859384724068,
#               249.98006656900867, 216.61676144486935, 144.76740507780784, 221.06173860788428, 173.5, 180.0999984741211,
#               139.5, 190.0999984741211, 207.91689978944407, 237.0190670093871, 139.77793287440193, 284.7367783260421,
#               175.82214962009715, 311.80415622424437, 129.75056060325727, 364.9220675460983],
#              [201.89163307761822, 111.39988883503088, 190.86215524715473, 123.43568334345082, 208.89584419625584,
#               95.38093880116149, 0, 0, 0, 0, 215.92742758603816, 120.43147222481318, 155.78846067099602,
#               111.40831107230616, 229.94848317922637, 169.97330511649383, 125.7337161287067, 176.9859384724068,
#               250.98006656900867, 216.61676144486935, 145.76740507780784, 220.06173860788428, 173.5, 175.0999984741211,
#               141.5, 185.0999984741211, 203.91689978944407, 232.0190670093871, 144.77793287440193, 282.7367783260421,
#               179.82214962009715, 317.80415622424437, 129.75056060325727, 366.9220675460983],
#              [204.89163307761822, 108.39988883503088, 193.86215524715473, 120.43568334345082, 210.89584419625584,
#               91.38093880116149, 0, 0, 0, 0, 218.92742758603816, 117.43147222481318, 156.78846067099602,
#               108.40831107230616, 229.94848317922637, 169.97330511649383, 124.7337161287067, 175.9859384724068,
#               254.98006656900867, 214.61676144486935, 146.76740507780784, 220.06173860788428, 175.5, 173.0999984741211,
#               141.5, 182.0999984741211, 205.91689978944407, 234.0190670093871, 147.77793287440193, 281.7367783260421,
#               182.82214962009715, 319.80415622424437, 130.75056060325727, 364.9220675460983],
#              [207.89163307761822, 104.39988883503088, 198.86215524715473, 118.43568334345082, 214.89584419625584,
#               89.38093880116149, 0, 0, 0, 0, 221.92742758603816, 116.43147222481318, 159.78846067099602,
#               106.40831107230616, 230.94848317922637, 168.97330511649383, 126.7337161287067, 175.9859384724068,
#               255.98006656900867, 213.61676144486935, 146.76740507780784, 220.06173860788428, 178.5, 171.0999984741211,
#               143.5, 180.0999984741211, 205.91689978944407, 236.0190670093871, 149.77793287440193, 281.7367783260421,
#               184.82214962009715, 320.80415622424437, 132.75056060325727, 363.9220675460983],
#              [211.89163307761822, 103.39988883503088, 201.86215524715473, 116.43568334345082, 218.89584419625584,
#               88.38093880116149, 0, 0, 0, 0, 224.92742758603816, 115.43147222481318, 162.78846067099602,
#               104.40831107230616, 232.94848317922637, 167.97330511649383, 126.7337161287067, 175.9859384724068,
#               257.98006656900867, 211.61676144486935, 147.76740507780784, 222.06173860788428, 182.5, 173.0999984741211,
#               146.5, 180.0999984741211, 207.91689978944407, 234.0190670093871, 152.77793287440193, 282.7367783260421,
#               188.82214962009715, 325.80415622424437, 134.75056060325727, 363.9220675460983],
#              [215.89163307761822, 103.39988883503088, 205.86215524715473, 116.43568334345082, 222.89584419625584,
#               85.38093880116149, 0, 0, 0, 0, 226.92742758603816, 110.43147222481318, 165.78846067099602,
#               102.40831107230616, 234.94848317922637, 167.97330511649383, 128.7337161287067, 176.9859384724068,
#               260.98006656900867, 211.61676144486935, 149.76740507780784, 223.06173860788428, 185.5, 174.0999984741211,
#               148.5, 180.0999984741211, 209.91689978944407, 231.0190670093871, 155.77793287440193, 283.7367783260421,
#               194.82214962009715, 328.80415622424437, 137.75056060325727, 362.9220675460983],
#              [222.89163307761822, 101.39988883503088, 210.86215524715473, 116.43568334345082, 227.89584419625584,
#               84.38093880116149, 0, 0, 0, 0, 231.92742758603816, 110.43147222481318, 170.78846067099602,
#               102.40831107230616, 240.94848317922637, 170.97330511649383, 130.7337161287067, 175.9859384724068,
#               262.98006656900867, 210.61676144486935, 152.76740507780784, 231.06173860788428, 186.5, 175.0999984741211,
#               149.5, 180.0999984741211, 212.91689978944407, 227.0190670093871, 159.77793287440193, 285.7367783260421,
#               198.82214962009715, 330.80415622424437, 139.75056060325727, 359.9220675460983],
#              [227.89163307761822, 99.39988883503088, 215.86215524715473, 114.43568334345082, 233.89584419625584,
#               82.38093880116149, 0, 0, 0, 0, 239.92742758603816, 110.43147222481318, 175.78846067099602,
#               102.40831107230616, 244.94848317922637, 171.97330511649383, 133.7337161287067, 170.9859384724068,
#               268.98006656900867, 209.61676144486935, 153.76740507780784, 232.06173860788428, 195.5, 176.0999984741211,
#               153.5, 180.0999984741211, 216.91689978944407, 235.0190670093871, 163.77793287440193, 287.7367783260421,
#               203.82214962009715, 336.80415622424437, 143.75056060325727, 356.9220675460983],
#              [235.89163307761822, 98.39988883503088, 221.86215524715473, 114.43568334345082, 240.89584419625584,
#               81.38093880116149, 0, 0, 0, 0, 245.92742758603816, 110.43147222481318, 180.78846067099602,
#               101.40831107230616, 250.94848317922637, 172.97330511649383, 137.7337161287067, 170.9859384724068,
#               273.98006656900867, 210.61676144486935, 155.76740507780784, 232.06173860788428, 199.5, 178.0999984741211,
#               158.5, 180.0999984741211, 225.91689978944407, 245.0190670093871, 169.77793287440193, 290.7367783260421,
#               208.82214962009715, 341.80415622424437, 148.75056060325727, 356.9220675460983],
#              [241.89163307761822, 96.39988883503088, 228.86215524715473, 113.43568334345082, 248.89584419625584,
#               76.38093880116149, 0, 0, 0, 0, 251.92742758603816, 110.43147222481318, 187.78846067099602,
#               100.40831107230616, 256.9484831792264, 173.97330511649383, 143.7337161287067, 168.9859384724068,
#               277.98006656900867, 211.61676144486935, 159.76740507780784, 234.06173860788428, 207.5, 181.0999984741211,
#               163.5, 180.0999984741211, 229.91689978944407, 247.0190670093871, 176.77793287440193, 291.7367783260421,
#               215.82214962009715, 351.80415622424437, 154.75056060325727, 356.9220675460983],
#              [251.89163307761822, 94.39988883503088, 237.86215524715473, 112.43568334345082, 259.89584419625584,
#               74.38093880116149, 0, 0, 0, 0, 263.92742758603816, 110.43147222481318, 197.78846067099602,
#               98.40831107230616, 264.9484831792264, 173.97330511649383, 148.7337161287067, 167.9859384724068,
#               282.98006656900867, 212.61676144486935, 166.76740507780784, 234.06173860788428, 214.5, 181.0999984741211,
#               170.5, 180.0999984741211, 235.91689978944407, 253.01906700938707, 184.77793287440193, 291.7367783260421,
#               220.82214962009715, 355.80415622424437, 161.75056060325727, 354.9220675460983],
#              [261.8916330776182, 90.39988883503088, 248.86215524715473, 111.43568334345082, 267.89584419625584,
#               71.38093880116149, 0, 0, 0, 0, 274.92742758603816, 108.43147222481318, 206.78846067099602,
#               97.40831107230616, 271.9484831792264, 174.97330511649383, 156.7337161287067, 169.9859384724068,
#               289.98006656900867, 212.61676144486935, 176.76740507780784, 236.06173860788428, 224.5, 184.0999984741211,
#               176.5, 180.0999984741211, 251.91689978944407, 267.0190670093871, 195.77793287440193, 291.7367783260421,
#               228.82214962009715, 364.80415622424437, 168.75056060325727, 350.9220675460983],
#              [272.8916330776182, 87.39988883503088, 256.86215524715476, 108.43568334345082, 276.89584419625584,
#               68.38093880116149, 0, 0, 0, 0, 280.92742758603816, 107.43147222481318, 216.78846067099602,
#               93.40831107230616, 283.9484831792264, 174.97330511649383, 163.7337161287067, 170.9859384724068,
#               297.98006656900867, 214.61676144486935, 181.76740507780784, 238.06173860788428, 234.5, 184.0999984741211,
#               185.5, 180.0999984741211, 257.9168997894441, 271.0190670093871, 201.77793287440193, 291.7367783260421,
#               235.82214962009715, 375.80415622424437, 179.75056060325727, 348.9220675460983]], [
#              [294.5, 158.0999984741211, 279.5, 165.0999984741211, 302.5, 144.5999984741211, 0, 0, 0, 0, 296.5,
#               170.0999984741211, 255.5, 148.0999984741211, 301.5, 191.0999984741211, 229.5, 183.0999984741211, 308.5,
#               225.0999984741211, 0, 0, 267.5, 220.0999984741211, 241.5, 216.0999984741211, 286.5, 270.10000014305115,
#               245.5, 264.10000014305115, 253.5, 317.10000014305115, 231.5, 335.10000014305115],
#              [300.5, 157.0999984741211, 286.5, 163.0999984741211, 306.5, 142.5999984741211, 0, 0, 0, 0, 303.5,
#               168.0999984741211, 263.5, 144.0999984741211, 307.5, 193.0999984741211, 233.5, 183.0999984741211, 313.5,
#               224.0999984741211, 0, 0, 268.5, 218.0999984741211, 244.5, 216.0999984741211, 287.5, 270.10000014305115,
#               246.5, 264.10000014305115, 258.5, 318.10000014305115, 233.5, 335.10000014305115],
#              [303.5, 151.0999984741211, 290.5, 162.0999984741211, 310.5, 134.5999984741211, 0, 0, 0, 0, 305.5,
#               166.0999984741211, 265.5, 142.0999984741211, 310.5, 193.0999984741211, 239.5, 180.0999984741211, 319.5,
#               223.0999984741211, 0, 0, 269.5, 213.0999984741211, 247.5, 210.0999984741211, 290.5, 270.10000014305115,
#               248.5, 268.10000014305115, 261.5, 319.10000014305115, 237.5, 335.10000014305115],
#              [306.5, 149.0999984741211, 294.5, 160.0999984741211, 313.5, 133.5999984741211, 0, 0, 0, 0, 308.5,
#               160.0999984741211, 268.5, 142.0999984741211, 312.5, 191.0999984741211, 239.5, 182.0999984741211, 325.5,
#               223.0999984741211, 0, 0, 272.5, 211.0999984741211, 249.5, 210.0999984741211, 292.5, 270.10000014305115,
#               250.5, 268.10000014305115, 267.5, 319.10000014305115, 240.5, 335.10000014305115],
#              [310.5, 148.0999984741211, 295.5, 158.0999984741211, 315.5, 130.5999984741211, 0, 0, 0, 0, 310.5,
#               160.0999984741211, 269.5, 142.0999984741211, 316.5, 191.0999984741211, 240.5, 184.0999984741211, 329.5,
#               223.0999984741211, 0, 0, 276.5, 208.0999984741211, 246.5, 206.0999984741211, 293.5, 270.10000014305115,
#               248.5, 270.10000014305115, 271.5, 324.10000014305115, 242.5, 335.10000014305115],
#              [313.5, 146.0999984741211, 297.5, 157.0999984741211, 317.5, 129.5999984741211, 0, 0, 0, 0, 313.5,
#               159.0999984741211, 271.5, 143.0999984741211, 318.5, 191.0999984741211, 246.5, 184.0999984741211, 334.5,
#               223.0999984741211, 0, 0, 275.5, 209.0999984741211, 248.5, 204.0999984741211, 296.5, 270.10000014305115,
#               248.5, 272.10000014305115, 278.5, 327.10000014305115, 244.5, 335.10000014305115],
#              [317.5, 144.0999984741211, 301.5, 155.0999984741211, 320.5, 127.5999984741211, 0, 0, 0, 0, 315.5,
#               157.0999984741211, 275.5, 143.0999984741211, 325.5, 191.0999984741211, 251.5, 180.0999984741211, 339.5,
#               222.0999984741211, 278.5, 223.0999984741211, 276.5, 208.0999984741211, 248.5, 202.0999984741211, 300.5,
#               270.10000014305115, 252.5, 273.10000014305115, 281.5, 326.10000014305115, 249.5, 332.10000014305115],
#              [320.5, 143.0999984741211, 306.5, 154.0999984741211, 324.5, 124.5999984741211, 0, 0, 0, 0, 318.5,
#               157.0999984741211, 278.5, 142.0999984741211, 328.5, 189.0999984741211, 257.5, 180.0999984741211, 344.5,
#               221.0999984741211, 279.5, 221.0999984741211, 280.5, 207.0999984741211, 251.5, 201.0999984741211, 300.5,
#               269.10000014305115, 255.5, 273.10000014305115, 287.5, 326.10000014305115, 251.5, 332.10000014305115],
#              [321.5, 141.0999984741211, 308.5, 152.0999984741211, 328.5, 119.5999984741211, 0, 0, 0, 0, 321.5,
#               154.0999984741211, 281.5, 140.0999984741211, 331.5, 189.0999984741211, 260.5, 178.0999984741211, 349.5,
#               221.0999984741211, 285.5, 221.0999984741211, 284.5, 204.0999984741211, 255.5, 203.0999984741211, 300.5,
#               271.10000014305115, 260.5, 275.10000014305115, 291.5, 326.10000014305115, 253.5, 332.10000014305115],
#              [326.5, 139.0999984741211, 311.5, 152.0999984741211, 330.5, 117.5999984741211, 0, 0, 0, 0, 324.5,
#               154.0999984741211, 285.5, 136.0999984741211, 336.5, 186.0999984741211, 265.5, 177.0999984741211, 358.5,
#               221.0999984741211, 290.5, 223.0999984741211, 291.5, 205.0999984741211, 259.5, 200.0999984741211, 302.5,
#               273.10000014305115, 265.5, 275.10000014305115, 298.5, 322.10000014305115, 256.5, 332.10000014305115],
#              [332.5, 136.0999984741211, 315.5, 150.0999984741211, 336.5, 115.5999984741211, 0, 0, 0, 0, 329.5,
#               152.0999984741211, 290.5, 135.0999984741211, 341.5, 187.0999984741211, 271.5, 180.0999984741211, 362.5,
#               221.0999984741211, 297.5, 225.0999984741211, 294.5, 204.0999984741211, 260.5, 198.0999984741211, 306.5,
#               273.10000014305115, 271.5, 277.10000014305115, 303.5, 323.10000014305115, 263.5, 337.10000014305115],
#              [336.5, 134.0999984741211, 320.5, 148.0999984741211, 340.5, 111.5999984741211, 0, 0, 0, 0, 334.5,
#               150.0999984741211, 293.5, 135.0999984741211, 349.5, 188.0999984741211, 278.5, 182.0999984741211, 366.5,
#               220.0999984741211, 304.5, 226.0999984741211, 294.5, 207.0999984741211, 266.5, 198.0999984741211, 313.5,
#               273.10000014305115, 275.5, 277.10000014305115, 311.5, 321.10000014305115, 266.5, 340.10000014305115],
#              [341.5, 132.0999984741211, 326.5, 148.0999984741211, 344.5, 109.5999984741211, 0, 0, 0, 0, 340.5,
#               149.0999984741211, 301.5, 137.0999984741211, 354.5, 188.0999984741211, 282.5, 184.0999984741211, 371.5,
#               218.0999984741211, 311.5, 225.0999984741211, 299.5, 206.0999984741211, 270.5, 199.0999984741211, 322.5,
#               274.10000014305115, 286.5, 287.10000014305115, 314.5, 323.10000014305115, 267.5, 340.10000014305115],
#              [352.5, 129.0999984741211, 334.5, 147.0999984741211, 352.5, 108.5999984741211, 0, 0, 0, 0, 346.5,
#               150.0999984741211, 311.5, 137.0999984741211, 359.5, 189.0999984741211, 291.5, 184.0999984741211, 378.5,
#               219.0999984741211, 319.5, 224.0999984741211, 307.5, 206.0999984741211, 281.5, 203.0999984741211, 337.5,
#               270.10000014305115, 295.5, 292.10000014305115, 321.5, 325.10000014305115, 286.5, 337.10000014305115],
#              [358.5, 128.0999984741211, 345.5, 147.0999984741211, 361.5, 108.5999984741211, 0, 0, 0, 0, 356.5,
#               151.0999984741211, 320.5, 137.0999984741211, 366.5, 191.0999984741211, 299.5, 184.0999984741211, 388.5,
#               222.0999984741211, 331.5, 224.0999984741211, 315.5, 208.0999984741211, 288.5, 203.0999984741211, 342.5,
#               270.10000014305115, 301.5, 298.10000014305115, 332.5, 328.10000014305115, 288.5, 337.10000014305115],
#              [369.5, 130.0999984741211, 353.5, 146.0999984741211, 371.5, 108.5999984741211, 0, 0, 0, 0, 365.5,
#               151.0999984741211, 329.5, 137.0999984741211, 373.5, 193.0999984741211, 308.5, 186.0999984741211, 396.5,
#               221.0999984741211, 338.5, 227.0999984741211, 324.5, 208.0999984741211, 296.5, 203.0999984741211, 346.5,
#               272.10000014305115, 311.5, 296.10000014305115, 339.5, 328.10000014305115, 298.5, 337.10000014305115]]]
#     )
#     y.append(xxx.permute(1, 0, 2))
#     y.append(xxx.permute(1,0,2))
#     masks = torch.randint(low=0, high=2, size=(2, 17))
#     all_masks.append(masks)
#     all_masks.append(masks)
#     vis.visualizer_2D(poses=y, images_paths=[], name='2D_visualize_without_bg_image')
