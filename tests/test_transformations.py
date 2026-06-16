"""
Unit Tests for Spatial Transformations, Quaternions, and Kinematics
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
"""

import numpy as np
import pytest
from src.kinematics.transformations import (
    rot_x, rot_y, rot_z, euler_to_rot, rot_to_euler,
    rodrigues_rotation, quaternion_to_rot, rot_to_quaternion,
    quaternion_multiply, homogeneous_transform, invert_transform
)
from src.kinematics.planar_kinematics import Planar2RArm, Planar3RArm
from src.kinematics.dh_matrix import dh_standard_matrix, dh_chain


def test_rotation_matrices_orthonormality():
    for fn in [rot_x, rot_y, rot_z]:
        R = fn(0.75)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-7)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-7)


def test_euler_conversions_roundtrip():
    roll_in, pitch_in, yaw_in = 0.35, -0.42, 0.81
    R = euler_to_rot(roll_in, pitch_in, yaw_in)
    roll_out, pitch_out, yaw_out = rot_to_euler(R)

    assert np.isclose(roll_in, roll_out, atol=1e-5)
    assert np.isclose(pitch_in, pitch_out, atol=1e-5)
    assert np.isclose(yaw_in, yaw_out, atol=1e-5)


def test_rodrigues_and_quaternion_equivalence():
    axis = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])
    theta = 1.15

    R_rodrigues = rodrigues_rotation(axis, theta)
    q = np.array([axis[0] * np.sin(theta/2), axis[1] * np.sin(theta/2), axis[2] * np.sin(theta/2), np.cos(theta/2)])
    R_quat = quaternion_to_rot(q)
    q_recovered = rot_to_quaternion(R_rodrigues)

    assert np.allclose(R_rodrigues, R_quat, atol=1e-6)
    assert np.allclose(quaternion_to_rot(q_recovered