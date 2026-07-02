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
            # Wrap to [-pi, pi]
            th1 = (th1 + np.pi) % (2.0 * np.pi) - np.pi
            th2 = (th2 + np.pi) % (2.0 * np.pi) - np.pi
            solutions.append((float(th1), float(th2)))

        return solutions

    def jacobian(self, theta1: float, theta2: float) -> np.ndarray:
        """Computes 2x2 Jacobian matrix: [dx/dt; dy/dt] = J * [dtheta1/dt; dtheta2/dt]."""
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        s12 = np.sin(theta1 + theta2)
        c12 = np.cos(theta1 + theta2)

        return np.array([
            [-self.l1 * s1 - self.l2 * s12, -self.l2 * s12],
            [ self.l1 * c1 + self.l2 * c12,  self.l2 * c12]
        ], dtype=np.float64)


class Planar3RArm:
    """Analytical kinematics for a 3-link planar revolute manipulator with end-effector orientation."""
    def __init__(self, l1: float = 1.0, l2: float = 1.0, l3: float = 0.5):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.l3 = float(l3)
        self.arm_2r = Planar2RArm(l1=l1, l2=l2)

    def forward_kinematics(self, theta1: float, theta2: float, theta3: float) -> Tuple[float, float, float]:
        """Calculates (x, y, phi) where phi = theta1 + theta2 + theta3 is tool orientation."""
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2) + self.l3 * np.cos(theta1 + theta2 + theta3)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2) + self.l3 * np.sin(theta1 + theta2 + theta3)
        phi = (theta1 + theta2 + theta3 + np.pi) % (2.0 * np.