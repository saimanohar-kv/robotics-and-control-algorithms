"""
Unit Tests for Control Algorithms: PID, LQR, and Feedforward
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
"""

import numpy as np
import pytest
from src.control.pid_controller import DiscretePIDController
from src.control.lqr_regulator import DiscreteLQRRegulator
from src.control.feedforward import FeedforwardTorqueCompensator


def test_pid_clamping_antiwindup():
    pid = DiscretePIDController(
        kp=10.0, ki=5.0, kd=1.0, dt=0.01,
        output_min=-5.0, output_max=5.0
    )

    # Apply large step error to saturate output
    setpoint = 100.0
    measurement = 0.0

    for _ in range(50):
        u = pid.update(setpoint, measurement)
        assert u <= 5.0 + 1e-9
        assert u >= -5.0 - 1e-9

    # Internal integral must be clamped and not runaway
    assert abs(pid.integral) <= 10.0


def test_pid_convergence_linear_plant():
    # 1st-order plant: x[k+1] = 0.9 * x[k] + 0.1 * u[k]
    pid = DiscretePIDController(kp=5.0, ki=6.0, kd=0.1, dt=0.02, output_min=-20.0, output_max=20.0)
    x = 0.0
    target = 1.0

    for _ in range(300):
        u = pid.update(target, x)
        x = 0.9 * x + 0.1 * u

    assert np.isclose(x, target, atol=0.01)


def test_l