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
    Extracts Eule