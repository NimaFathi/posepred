import numpy as np
import torch
import cv2

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


def xyz_to_spherical(inputs):
    """
    Convert cartesian representation to spherical representation.
    Args:
      inputs -- cartesian coordinates. (..., 3)
    
    Returns:
      out -- spherical coordinate. (..., 3)
    """
    
    rho = torch.norm(inputs, p=2, dim=-1)
    theta = torch.arctan(inputs[..., 2] / (inputs[..., 0] + 1e-8)).unsqueeze(-1)
    tol = 0
    theta[inputs[..., 0] < tol] = theta[inputs[..., 0] < tol] + torch.pi
    phi = torch.arccos(inputs[..., 1] / (rho + 1e-8)).unsqueeze(-1)
    rho = rho.unsqueeze(-1)
    out = torch.cat([rho, theta, phi], dim=-1)
    out[out.isnan()] = 0

    return out

def spherical_to_xyz(self, inputs):
    """
    Convert cartesian representation to spherical representation.
    Args:
      inputs -- spherical coordinates. (..., 3)
    
    Returns:
      out -- cartesian coordinate. (..., 3)
    """
    
    x = inputs[..., 0] * torch.sin(inputs[..., 2]) * torch.cos(inputs[..., 1])
    y = inputs[..., 0] * torch.sin(inputs[..., 2]) * torch.sin(inputs[..., 1])
    z = inputs[..., 0] * torch.cos(inputs[..., 2])
    x, y, z = x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)

    return torch.cat([x, z, y], dim=-1)

def sig5(p:torch.Tensor, x:torch.Tensor):
    """
    Arguments:
        p -- sig5 parameters. shape: ..., 5
        x -- input of sig5 function. shape: ... 
    Return:
        output -- output of sig5 function. 
    """
    assert p.shape[-1] == 5
    if len(p.shape) == 1: p = p.reshape(1, -1)
    p_shape = p.shape 
    x_shape = x.shape 

    p = p.reshape(-1, 5) # 20, 5
    x = x.reshape(1, -1) # 1, 23
    
    p1 = p[:, 0].unsqueeze(1) # 20, 1
    p2 = p[:, 1].unsqueeze(1)
    p3 = p[:, 2].unsqueeze(1)
    p4 = p[:, 3].unsqueeze(1)
    p5 = p[:, 4].unsqueeze(1)

    c = 2*p3*p5/torch.abs(p3+p5) # 20, 1
    f = 1/(1+torch.exp(-c*(p4-x))) # 20, 23
    g = torch.exp(p3*(p4-x)) # 20, 23
    h = torch.exp(p5*(p4-x)) # 20, 23
    output = (p1+(p2/(1+f*g+(1-f)*h))) # 20, 23
    output = output.reshape(*p_shape[:-1], *x_shape)
    return output


def sigstar(p:torch.Tensor, x:torch.Tensor):
    """
    Arguments:
        p -- sig* parameters. shape: ..., 3
        x -- input of sig* function. shape: ... 
    Return:
        output -- output of sig* function. 
    """
    assert p.shape[-1] == 3
    if len(p.shape) == 1: p = p.reshape(1, -1)
    p_shape = p.shape 
    x_shape = x.shape 

    p = p.reshape(-1, 3) # 20, 3
    x = x.reshape(1, -1) # 1, 23
    
    x0 = p[:, 0].unsqueeze(1) # 20, 1
    k = p[:, 1].unsqueeze(1)
    L = p[:, 2].unsqueeze(1)

    output = L / (1 + torch.exp(-k * (x - x0))) # 20, 23
    output = output.reshape(*p_shape[:-1], *x_shape) # 
    return output


if __name__ == '__main__':
  p = torch.rand(3, 4, 3)
  x = torch.rand(2, 6, 3, 5)
  print(sigstar(p, x).shape)