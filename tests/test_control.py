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


def test_lqr_regulator_stability():
    # Discrete double integrator (e.g. mass in 1D):
    # x = [pos, vel]^T, u = force
    dt = 0.05
    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
    B = np.array([[0.5 * dt**2],
                  [dt]])
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])

    lqr = DiscreteLQRRegulator(A, B, Q, R)
    assert lqr.is_closed_loop_stable()

    # Simulate regulation to origin from x0 = [2.0, -1.0]
    x = np.array([2.0, -1.0])
    for _ in range(150):
        u = lqr.compute_control(x)
        x = A @ x + B @ u

    assert np.linalg.norm(x) < 1e-3


def test_feedforward_gravity_compensation():
    masses = np.array([2.0, 1.5])
    lengths = np.array([0.5, 0.4])
    ff = FeedforwardTorqueCompensator(masses, lengths)

    # Arm horizontal: q = [0, 0] -> Max gravity torque
    q_horiz = np.array([0.0, 0.0])
    tau_g_horiz = ff.compute_gravity_compensation(q_horiz)

    # Arm hanging straight down: q = [-pi/2, 0] -> Zero gravity torque
    q_vert = np.array([-np.pi / 2.0, 0.0])
    tau_g_vert = ff.compute_gravity_compensation(q_vert)

    assert tau_g_horiz[0] > 0.0
    assert np.isclose(tau_g_vert[0], 0.0, atol=1e-5)
