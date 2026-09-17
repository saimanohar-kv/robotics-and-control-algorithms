"""
Trapezoidal (Bang-Coast-Bang) Velocity Profile Generator
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import Tuple
import numpy as np


class TrapezoidalVelocityProfile:
    """
    Symmetric trapezoidal velocity trajectory:
    Computes position, velocity, and acceleration for rest-to-rest 1D point-to-point motion.
    Automatically handles triangular degenerate profile if max velocity cannot be reached.
    """
    def __init__(self, max_velocity: float, max_acceleration: float):
        self.v_max = float(abs(max_velocity))
        self.a_max = float(abs(max_acceleration))

    def plan(self, q0: float, qf: float) -> Tuple[float, float, float, float]:
        """
        Calculates timing intervals (ta, tc, td, T_total).
        ta: acceleration phase duration
        tc: constant cruise velocity phase duration
        td: deceleration phase duration
        T_total: total trajectory duration
        """
        dist = qf - q0
        self.direction = 1.0 if dist >= 0.0 else -1.0
        abs_dist = abs(dist)

        # Min distance required to reach v_max and brake to stop
        # d_crit = 2 * (0.5 * a_max * ta^2) = v_max^2 / a_max
        d_crit = (self.v_max ** 2) / self.a_max

        if abs_dist >= d_crit:
            # Full trapezoid: accelerates to v_max, coasts, then decelerates
            self.ta = self.v_max / self.a_max
            self.td = self.ta
            d_cruise = abs_dist - d_crit
            self.tc = d_cruise / self.v_max
            self.v_peak = self.v_max
        else:
            # Triangular profile: peak velocity is lower than v_max
            self.v_peak = np.sqrt(abs_dist * self.a_max)
            self.ta = self.v_peak / self.a_max
            self.td = self.ta
            self.tc = 0.0

        self.q0 = float(q0)
        self.qf = float(qf)
        self.T = self.ta + self.tc + self.td
        return self.ta, self.tc, self.td, self.T

    def evaluate(self, t: float) -> Tuple[float, float, float]:
        """
        Evaluates position s(t), velocity v(t), and acceleration a(t) at time t.
        """
        if t <= 0.0:
            return self.q0, 0.0, 0.0
        elif t >= self.T:
            return self.qf, 0.0, 0.0

        if t <= self.ta:
            # Acceleration phase
            s = self.q0 + self.direction * 0.5 * self.a_max * (t ** 2)
            v = self.direction * self.a_max * t
            a = self.direction * self.a_max
        elif t <= (self.ta + self.tc):
            # Cruise phase
            s_acc = 0.5 * self.a_max * (self.ta ** 2)
            dt_cruise = t - self.ta
            s = self.q0 + self.direction * (s_acc + self.v_peak * dt_cruise)
            v = self.direction * self.v_peak
            a = 0.0
        else:
            # Deceleration phase
            dt_dec = t - (self.ta + self.tc)
            s_acc = 0.5 * self.a_max * (self.ta ** 2)
            s_cruise = self.v_peak * self.tc
            s_dec = self.v_peak * dt_dec - 0.5 * self.a_max * (dt_dec ** 2)
            s = self.q0 + self.direction * (s_acc + s_cruise + s_dec)
            v = self.direction * (self.v_peak - self.a_max * dt_dec)
            a = -self.direction * self.a_max

        return float(s), float(v), float(a)
