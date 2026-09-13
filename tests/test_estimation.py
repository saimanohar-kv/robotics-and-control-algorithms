"""
Unit Tests for Estimation: Linear Kalman Filter, EKF, and Complementary Filter
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
"""

import numpy as np
import pytest
from src.estimation.kalman_filter import LinearKalmanFilter
from src.estimation.extended_kalman_filter import RobotPoseEKF
from src.estimation.complementary_filter import ComplementaryOrientationFilter


def test_linear_kalman_filter_tracking():
    # 1D Constant velocity model: x = [pos, vel]^T
    dt = 0.1
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([0.01, 0.01])
    R = np.array([[0.5]])

    kf = LinearKalmanFilter(F, H, Q, R, x0=np.array([0.0, 0.0]))

    true_velocity = 2.0
    true_pos = 0.0
    estimates = []

    np.random.seed(42)
    for _ in range(50):
        true_pos += true_velocity * dt
        # Noisy measurement
        z = true_pos + np.random.normal(0, np.sqrt(0.5))
        kf.predict()
        x_est, P_est = kf.update(np.array([z]))
        estimates.append(x_est)

    final_est = estimates[-1]
    assert np.isclose(final_est[0], true_pos, atol=0.8)
    assert np.isclose(final_est[1], true_velocity, atol=0.5)


def test_extended_kalman_filter_circular_trajectory():
    ekf = RobotPoseEKF(dt=0.1)
    v = 1.0       # 1 m/s
    omega = 0.2   # 0.2 rad/s

    np.random.seed(42)
    for _ in range(100):
        # Predict step
        pred = ekf.predict(v, omega)
        
        # Simulated measurement with noise
        z = pred + np.random.normal(0, 0.02, size=3)
        state_est, P_est = ekf.update(z)

    # State covariance must remain positive-definite and shrink
    assert np.all(np.linalg.eigvals(P_est) > 0.0)
    assert np.trace(P_est) < 0.5


def test_complementary_filter_attitude_fusion():
    filt = ComplementaryOrientationFilter(alpha=0.95, dt=0.01)

    # Static gravity vector pointing along Z: no tilt
    accel_zero = np.array([0.0, 0.0, 9.81])
    gyro_zero = np.array([0.0, 0.0, 0.0])

    roll, pitch = filt.update(accel_zero, gyro_zero)
    assert np.isclose(roll, 0.0, atol=1e-3)
    assert np.isclose(pitch, 0.0, atol=1e-3)

    # Roll tilt 30 deg (0.5236 rad)
    # Gravity components: ay = g * sin(30), az = g * cos(30)
    angle_target = np.radians(30.0)
    accel_tilt = np.array([0.0, 9.81 * np.sin(angle_target), 9.81 * np.cos(angle_target)])

    for _ in range(150):
        roll, pitch = filt.update(accel_tilt, gyro_zero)

    assert np.isclose(roll, angle_target, atol=0.02)
    assert np.isclose(pitch, 0.0, atol=0.02)
