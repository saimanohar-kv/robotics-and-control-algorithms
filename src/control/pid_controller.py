"""
Discrete PID Controller with Low-Pass Derivative Filtering and Anti-Windup Clamping
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import Optional
import numpy as np


class DiscretePIDController:
    """
    Robust Discrete PID Controller:
    - Trapezoidal integration (Tustin) for low phase lag
    - First-order derivative low-pass filter to attenuate high-frequency noise
    - Conditional anti-windup clamping to prevent integrator runaway
    """
    def __init__(self,
                 kp: float,
                 ki: float,
                 kd: float,
                 dt: float,
                 output_min: float = -float('inf'),
                 output_max: float = float('inf'),
                 filter_time_constant: float = 0.01):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.tau = float(filter_time_constant)

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.prev_measurement: Optional[float] = None

    def reset(self):
        """Resets controller internal states."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.prev_measurement = None

    def update(self, setpoint: float, measurement: float) -> float:
        """
        Computes control output u[k] given setpoint and current measurement.
        """
        error = setpoint - measurement

        # 1. Proportional Term
        p_term = self.kp * error

        # 2. Integral Term with Trapezoidal approximation
        int_candidate = self.integral + 0.5 * self.ki * self.dt * (error + self.prev_error)

        # 3. Filtered Derivative Term
        # D[k] = (tau / (tau + dt)) * D[k-1] + (kd / (tau + dt)) * (e[k] - e[k-1])
        alpha = self.tau / (self.tau + self.dt)
        d_term = alpha * self.prev_derivative + (self.kd / (self.tau + self.dt)) * (error - self.prev_error)

        # Candidate output without windup check
        u_raw = p_term + int_candidate + d_term

        # 4. Anti-Windup Clamping
        # Only integrate if output is not saturated, or if error pushes output back into bounds
        if self.output_min <= u_raw <= self.output_max:
            self.integral = int_candidate
        elif u_raw > self.output_max and error < 0.0:
            self.integral = int_candidate
        elif u_raw < self.output_min and error > 0.0:
            self.integral 