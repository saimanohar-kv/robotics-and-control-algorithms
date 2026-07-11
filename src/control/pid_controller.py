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

    d