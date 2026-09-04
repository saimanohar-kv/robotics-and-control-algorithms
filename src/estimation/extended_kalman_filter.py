"""
Extended Kalman Filter (EKF) for Non-Linear Mobile Robot Pose Estimation
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import Tuple
import numpy as np


class RobotPoseEKF:
    """
    Non-linear Extended Kalman Filter for a differential-drive / unicycle robot.
    State: x = [pos_x, pos_y, theta]^T
    Control: u = [v, omega]^T (linear velocity, angular velocity)
    Measurement: z = [meas_x, meas_y, meas_theta]^T (e.g. GPS / Lidar scan matching / Landmark)
    """
    def __init__(self,
                 dt: float = 0.05,
                 q_pos: float = 0.01,
                 q_theta: float = 0.02,
                 r_pos: float = 0.05,
                 r_theta: float = 0.05):
        self.dt = float(dt)
        self.state = np.zeros(3, dtype=np.float64)  # [x, y, theta]
        self.P = np.eye(3, dtype=np.float64) * 0.1

        # Process noise covariance
        self.Q = np.diag([q_pos**2, q_pos**2, q_theta**2])
        # Measurement noise covariance
        self.R = np.diag([r_pos**2, r_pos**2, r_theta**2])

    def predict(self, v: float, omega: float) -> np.ndarray:
        """
        Non-linear motion propagation:
            x_(k+1) = x_k + v * dt * cos(theta_k + 0.5 * omega * dt)
            y_(k+1) = y_k + v * dt * sin(theta_k + 0.5 * omega * dt)
            theta_(k+1) = theta_k + omega * dt
        """
        th = self.state[2]
        dth = omega * self.dt
        th_mid = th + 0.5 * dth

        # State transition
        self.state[0] += v * self.dt * np.cos(th_mid)
        self.state[1] += v * self.dt * np.sin(th_mid)
        self.state[2] = (self.state[2] + dth + np.pi) % (2.0 * np.pi) - np.pi

        # Motion model Jacobian F_x = df / dx
        F_x = np.eye(3, dtype=np.float64)
        F_x[0, 2] =