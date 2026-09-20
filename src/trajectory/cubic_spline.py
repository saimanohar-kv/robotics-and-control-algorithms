"""
Piecewise Cubic Spline Trajectory Interpolator
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import Sequence, Tuple
import numpy as np


class NaturalCubicSpline1D:
    """
    Natural Cubic Spline interpolator guaranteeing C2 continuity
    (continuous position, velocity, and acceleration) through waypoints (t_i, y_i).
    """
    def __init__(self, time_points: Sequence[float], value_points: Sequence[float]):
        t = np.asarray(time_points, dtype=np.float64)
        y = np.asarray(value_points, dtype=np.float64)

        if len(t) < 2:
            raise ValueError("At least 2 points are required for spline interpolation.")
        if not np.all(np.diff(t) > 0):
            raise ValueError("Time points must be strictly monotonically increasing.")

        self.t = t
        self.y = y
        self.n = len(t)

        self._build_spline()

    def _build_spline(self):
        n = self.n
        h = np.diff(self.t)
        
        # Tridiagonal system matrix for second derivatives m_i
        A = np.zeros((n, n), dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)

        # Natural boundary condition: m_0 = 0, m_(n-1) = 0
        A[0, 0] = 1.0
        A[n - 1, n - 1] = 1.0

        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            b[i] = 6.0 * ((self.y[i + 1] - self.y[i]) / h[i] - (self.y[i] - self.y[i - 1]) / h[i - 1])

        # Solve for second derivatives (curvatures)
        self.m = np.linalg.solve(A, b)

    def evaluate(self, t_query: float) -> Tuple[float, float, float]:
        """
        Evaluates position s(t), velocity v(t), and acceleration a(t).
        """
        if t_query <= self.t[0]:
            return float(self.y[0]), 0.0, 0.0
        elif t_query >= self.t[-1]:
            return float(self.y[-1]), 0.0, 0.0

        # Find segment index i such that t[i] <= t_query < t[i+1]
        i = np.searchsorted(self.t, t_query) - 1
        i = max(0, min(i, self.n - 2))

        dt = self.t[i + 1] - self.t[i]
        d1 = t_query - self.t[i]
        d2 = self.t[i + 1] - t_query

        # Position evaluation
        pos = (self.m[i] * (d2 ** 3) + self.m[i + 1] * (d1 ** 3)) / (6.0 * dt) + \
              (self.y[i] / dt - self.m[i] * dt / 6.0) * d2 + \
              (self.y[i + 1] / dt - self.m[i + 1] * dt / 6.0) * d1

        # Velocity evaluation
        vel = (-self.m[i] * (d2 ** 2) + self.m[i + 1] * (d1 ** 2)) / (2.0 * dt) - \
              (self.y[i] / dt - self.m[i] * dt / 6.0) + \
              (self.y[i + 1] / dt - self.m[i + 1] * dt / 6.0)

        # Acceleration evaluation
        acc = (self.m[i] * d2 + self.m[i + 1] * d1) / dt

        return float(pos), float(vel), float(acc)
