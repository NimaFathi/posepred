import numpy as np
import torch
import cv2
from torch.autograd.variable import Variable

def pose_from_vel(velocity, last_obs_pose, stay_in_frame=False):
    device = 'cuda' if velocity.is_cuda else 'cpu'
    pose = torch.zeros_like(velocity).to(device)
    last_obs_pose_ = last_obs_pose

    for i in range(velocity.shape[-2]):
        pose[..., i, :] = last_obs_pose_ + velocity[..., i, :]
        last_obs_pose_ = pose[..., i, :]

    if stay_in_frame:
        for i in range(velocity.shape[-1]):
            pose[..., i] = torch.min(pose[..., i], 1920 * torch.ones_like(pose.shape[:-1]).to('cuda'))
            pose[..., i] = torch.max(pose[..., i], torch.zeros_like(pose.shape[:-1]).to('cuda'))

    return pose


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def get_binary(src, device):
    zero = torch.zeros_like(src).to(device)
    one = torch.ones_like(src).to(device)
    return torch.where(src > 0.5, one, zero)


def dict_to_device(src, device):
    out = dict()
    for key, value in src.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.clone().to(device)
        else:
            out[key] = value
    return out


def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)
    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)



def rotmat_to_expmap(action_sequence):
  """Convert exponential maps to rotmats.

  Args:
    action_sequence: [n_samples, n_joints, 9]
  Returns:
    Rotation matrices for exponenital maps [n_samples, n_joints, 3].
  """
  n_samples, n_joints, _ = action_sequence.shape
  rotmat = np.reshape(action_sequence, [n_samples*n_joints, 3, 3])
  # first three values are positions, so technically it's meaningless to convert them,
  # but we do it anyway because later we discard this values anywho
  expmap = np.zeros([n_samples*n_joints, 3, 1])
  for i in range(expmap.shape[0]):
    expmap[i] = cv2.Rodrigues(rotmat[i])[0]
  expmap = np.reshape(expmap, [n_samples, n_joints, 3])
  return expmap

def expmap_to_rotmat(action_sequence):
  """Convert exponential maps to rotmats.

  Args:
    action_sequence: [n_samples, n_joints, 3]
  Returns:
    Rotation matrices for exponenital maps [n_samples, n_joints, 9].
  """
  n_samples, n_joints, _ = action_sequence.shape
  expmap = np.reshape(action_sequence, [n_samples*n_joints, 1, 3])
  # first three values are positions, so technically it's meaningless to convert them,
  # but we do it anyway because later we discard this values anywho
  rotmats = np.zeros([n_samples*n_joints, 3, 3])
  for i in range(rotmats.shape[0]):
    rotmats[i] = cv2.Rodrigues(expmap[i])[0]
  rotmats = np.reshape(rotmats, [n_samples, n_joints, 3*3])
  return rotmats

def rotmat_to_euler(action_sequence):
  """Convert exponential maps to rotmats.

  Args:
    action_sequence: [n_samples, n_joints, 9]
  Returns:
    Euler angles for rotation maps given [n_samples, n_joints, 3].
  """
  n_samples, n_joints, _ = action_sequence.shape
  rotmats = np.reshape(action_sequence, [n_samples*n_joints, 3, 3])
  eulers = np.zeros([n_samples*n_joints, 3])
  for i in range(eulers.shape[0]):
    eulers[i] = rotmat2euler(rotmats[i])
  eulers = np.reshape(eulers, [n_samples, n_joints, 3])
  return eulers

def rotmat2euler(R):
  """Converts a rotation matrix to Euler angles.
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

  Args:
    R: a 3x3 rotation matrix

  Returns:
    eul: a 3x1 Euler angle representation of R
  """
  if R[0,2] >= 1 or R[0,2] <= -1:
    # special case values are out of bounds for arcsinc
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;
  else:
    E2 = -np.arcsin(R[0,2])
    E1 = np.arctan2(R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2(R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul

def expmap_to_euler(action_sequence):
  rotmats = expmap_to_rotmat(action_sequence)
  eulers = rotmat_to_euler(rotmats)
  return eulers

def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)


def normalize(in_tensor, mean, std):
    device = 'cuda' if in_tensor.is_cuda else 'cpu'
    bs, frame_n, feature_n = in_tensor.shape
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    keypoint_dim = mean.shape[0]
    assert mean.shape == std.shape
    assert feature_n % keypoint_dim == 0
    mean = mean.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)
    std = std.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)

    return (in_tensor - mean) / std


def denormalize(in_tensor, mean, std):
    device = 'cuda' if in_tensor.is_cuda else 'cpu'
    bs, frame_n, feature_n = in_tensor.shape
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    keypoint_dim = mean.shape[0]
    assert mean.shape == std.shape
    assert feature_n % keypoint_dim == 0
    mean = mean.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)
    std = std.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)

    return (in_tensor * std) + mean

def some_variables():
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

def fkl_torch(angles, parent, offset, rotInd, expmapInd):
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

def expmap_to_xyz_torch(expmap):
        """
        convert expmaps to joint locations
        :param expmap: N*99
        :return: N*32*3
        """
        expmap[:, 0:6] = 0
        parent, offset, rotInd, expmapInd = some_variables()
        xyz = fkl_torch(expmap, parent, offset, rotInd, expmapInd)
        return xyz

