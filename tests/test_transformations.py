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
    assert np.allclose(quaternion_to_rot(q_recovered), R_rodrigues, atol=1e-6)


def test_homogeneous_inverse():
    R = rot_z(0.5)
    p = np.array([1.2, -0.4, 2.1])
    T = homogeneous_transform(R, p)
    T_inv = invert_transform(T)

    assert np.allclose(T @ T_inv, np.eye(4), atol=1e-7)


def test_planar_2r_kinematics_roundtrip():
    arm = Planar2RArm(l1=1.2, l2=0.8)
    th1_orig, th2_orig = 0.45, -0.65
    x, y = arm.forward_kinematics(th1_orig, th2_orig)

    solutions = arm.inverse_kinematics(x, y)
    assert len(solutions) == 2

    # At least one solution must match original configuration
    matched = any(np.isclose(th1, th1_orig, atol=1e-5) and np.isclose(th2, th2_orig, atol=1e-5) for th1, th2 in solutions)
    assert matched

    # Verify both solutions achieve target coordinates
    for th1, th2 in solutions:
        x_calc, y_calc = arm.forward_kinematics(th1, th2)
        assert np.isclose(x, x_calc, atol=1e-5)
        assert np.isclose(y, y_calc, atol=1e-5)


def test_planar_3r_kinematics_roundtrip():
    arm3 = Planar3