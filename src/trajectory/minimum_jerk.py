"""
Minimum-Jerk Trajectory Generator (Flash & Hogan Model)
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import Tuple
import numpy as np


class MinimumJerkTrajectory:
    """
    Flash & Hogan (1985) Minimum Jerk Trajectory Generator.
    Minimizes the integral of squared jerk: J = integral_0^T (d^3 x / dt^3)^2 dt.
    Produces smooth, human-like point-to-point motion with zero boundary velocities and accelerations.
    """
    def __init__(self, start_pos: float, end_pos: float, duration: float):
        self.x0 = float(start_pos)
        self.xf = float(end_pos)
        self.T = float(duration)
        if self.T <= 0.0:
            raise ValueError(f"Duration T must be strictly positive, got {self.T}")

    def evaluate(self, t: float) -> Tuple[float, float, float]:
        """
        Evaluates position x(t), velocity v(t), and acceleration a(t) at time t in [0, T].
        Equations:
            tau = t / T
            x(t) = x0 + (xf - x0) * (10*tau^3 - 15*tau^4 + 6*tau^5)
            v(t) = ((xf - x0) / T) * (30*tau^2 - 60*tau^3 + 30*tau^4)
            a(t) = ((xf - x0) / T^2) * (60*tau - 180*tau^2 + 120*tau^3)
        """
        if t <= 0.0:
            return self.x0, 0.0, 0.0
        elif t >= self.T:
            return self.xf, 0.0, 0.0

        tau = t / self.T
        dx = self.xf - self.x0

        # Position
        pos = self.x0 + dx * (10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5))
        # Velocity
        vel = (dx / self.T) * (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4))
        # Acceleration
        acc = (dx / (self.T ** 2)) * (60.0 * tau - 180.0 * (tau ** 2) + 120.0 * (tau ** 3))

        return float(pos), float(vel), float(acc)
