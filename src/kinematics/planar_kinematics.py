"""
Forward and Inverse Kinematics for Planar 2R and 3R Robot Arms
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import List, Tuple
import numpy as np


class Planar2RArm:
    """Analytical kinematics for a 2-link planar revolute manipulator."""
    def __init__(self, l1: float = 1.0, l2: float = 1.0):
        self.l1 = float(l1)
        self.l2 = float(l2)

    def forward_kinematics(self, theta1: float, theta2: float) -> Tuple[float, float]:
        """Calculates (x, y) Cartesian position of end-effector."""
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
        return float(x), float(y)

    def inverse_kinematics(self, x: float, y: float) -> List[Tuple[float, float]]:
        """
        Analytical inverse kinematics returning both elbow-up and elbow-down configurations.
        Returns empty list if target point (x, y) is out of reachable workspace.
        """
        r2 = x**2 + y**2
        d = (r2 - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2)

        if np.abs(d) > 1.0 + 1e-9:
            return []  # Target out of reach

        d = np.clip(d, -1.0, 1.0)
        solutions = []

        # Elbow-up and Elbow-down
        for sign in [1.0, -1.0]:
            th2 = sign * np.arccos(d)
            th1 = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(th2), self.l1 + self.l2 * np.cos(th2))
            # Wrap to [-pi, 