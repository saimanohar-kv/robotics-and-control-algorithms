"""
Rigid Body Spatial Transformations in SO(3) and SE(3)
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import Tuple, Union
import numpy as np


def rot_x(theta: float) -> np.ndarray:
    """3x3 Rotation matrix around X-axis in SO(3)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c]
    ], dtype=np.float64)


def rot_y(theta: float) -> np.ndarray:
    """3x3 Rotation matrix around Y-axis in SO(3)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [  c, 0.0,   s],
        [0.0, 1.0, 0.0],
        [ -s, 0.0,   c]
    ], dtype=np.float64)


def rot_z(theta: float) -> np.ndarray:
    """3x3 Rotation matrix around Z-axis in SO(3)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def euler_to_rot(roll: float, pitch: float, yaw: float, order: str = 'xyz') -> np.ndarray:
    """
    Converts Euler angles to 3x3 rotation matrix in SO(3).
    Default convention: Extrinsic / Fixed-axis XYZ (Roll -> Pitch -> Yaw).
    """
    Rx = rot_x(roll)
    Ry = rot_y(pitch)
    Rz = rot_z(yaw)

    if order.lower() == 'xyz':
        return np.dot(Rz, np.dot(Ry, Rx))
    elif order.lower() == 'zyx':
        return np.dot(Rx, np.dot(Ry, Rz))
    else:
        raise ValueError(f"Unsupported Euler sequence: {order}")


def rot_to_euler(R: np.ndarray, order: str = 'xyz') -> Tuple[float, float, float]:
    """
    Extracts Euler angles (roll, pitch, yaw) from 3x3 rotation matrix.
    Handles gimbal lock singularities smoothly.
    """
    R = np.asarray(R, dtype=np.float64)
    if order.lower() == 'xyz':
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
        if np.abs(np.cos(pitch)) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            yaw = 0.0
        return float(roll), float(pitch), float(yaw)
    else:
        raise NotImplementedError(f"Extraction for order {order} not implemented.")


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Computes the 3x3 cross-product matrix [v]_x of a 3D vector v."""
    v = np.asarray(v, dtype=np.float64).flatten()
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def rodrigues_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Computes rotation matrix via Rodrigues' formula for rotation around unit axis k:
        R = I + sin(theta) * [k]_x + (1 - cos(theta)) * [k]_x^2
    """
    k = np.asarray(axis, dtype=np.float64).flatten()
    norm_k = np.linalg.norm(k)
    if norm_k < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = k / norm_k

    K = skew_symmetric(k)
    I = np.eye(3, dtype=np.float64)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * np.dot(K, K)


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalizes quaternion [qx, qy, qz, qw] to unit length."""
    q = np.asarray(q, dtype=np.float64).flatten()
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def quaternion_to_rot(q: np.ndarray) -> np.ndarray:
    """Converts unit quaternion [qx, qy, qz, qw] to 3x3 rotation matrix in SO(3)."""
    qx, qy, qz, qw = quaternion_normalize(q)
    return np.array([
        [1.0 - 2.0*(qy**2 + qz**2), 2.0*(qx*qy - qz*qw),       2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw),       1.0 - 2.0*(qx**2 + qz**2), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw),       2.0*(qy*qz + qx*qw),       1.0 - 2.0*(qx**2 + qy**2)]
    ], dtype=np.float64)


def rot_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Converts 3x3 rotation matrix to unit quaternion [qx, qy, qz, qw]
    using Shepperd's robust diagonal branch algorithm.
    """
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return quaternion_normalize(np.array([qx, qy, qz, qw], dtype=np.float64))


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product q = q1 * q2."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)


def homogeneous_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Constructs 4x4 homogeneous transformation matrix in SE(3)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=np.float64).flatten()
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverts 4x4 homogeneous transform T = [R p; 0 1] -> [R^T -R^T*p; 0 1]."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -np.dot(R.T, p)
    return T_inv
