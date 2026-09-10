"""
Complementary Filter for IMU Accelerometer and Gyroscope Attitude Fusion
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import Tuple
import numpy as np


class ComplementaryOrientationFilter:
    """
    Fuses high-frequency rate gyroscope data with low-frequency accelerometer gravity vector:
        angle_k = alpha * (angle_(k-1) + gyro_rate * dt) + (1 - alpha) * accel_angle
    """
    def __init__(self, alpha: float = 0.98, dt: float = 0.01):
        self.alpha = float(alpha)
        self.dt = float(dt)
        self.roll = 0.0   # Rotation around X axis (radians)
        self.pitch = 0.0  # Rotation around Y axis (radians)

    def update(self,
               accel: np.ndarray,
               gyro: np.ndarray) -> Tuple[float, float]:
        """
        accel: [ax, ay, az] in m/s^2 or g
        gyro: [gx, gy, gz] in rad/s
        Returns: (roll_rad, pitch_rad)
        """
        ax, ay, az = np.asarray(accel, dtype=np.float64).flatten()
        gx, gy, _ = np.asarray(gyro, dtype=np.float64).flatten()

        # Accelerometer static roll and pitch angles
        accel_roll = np.arctan2(ay, az)
        accel_pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

        # Complementary filter fusion
        self.roll = self.alpha * (self.roll + gx * self.dt) + (1.0 - self.alpha) * accel_roll
        self.pitch = self.alpha * (self.pitch + gy * self.dt) + (1.0 - self.alpha) * accel_pitch

        return float(self.roll), float(self.pitch)
