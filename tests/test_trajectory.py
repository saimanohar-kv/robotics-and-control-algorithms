"""
Unit Tests for Trajectory Generation: Trapezoidal, Cubic Spline, and Minimum Jerk
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
"""

import numpy as np
import pytest
from src.trajectory.trapezoidal_profile import TrapezoidalVelocityProfile
from src.trajectory.cubic_spline import NaturalCubicSpline1D
from src.trajectory.minimum_jerk import MinimumJerkTrajectory


def test_trapezoidal_profile_full_cruise():
    prof = TrapezoidalVelocityProfile(max_velocity=2.0, max_acceleration=4.0)
    q0, qf = 0.0, 10.0
    ta, tc, td, T = prof.plan(q0, qf)

    assert ta > 0.0
    assert tc > 0.0
    assert np.isclose(ta, td)

    # Initial state
    s0, v0, a0 = prof.evaluate(0.0)
    assert np.isclose(s0, q0)
    assert np.isclose(v0, 0.0)

    # Midpoint cruise velocity
    t_mid = ta + tc / 2.0
    _, v_mid, a_mid = prof.evaluate(t_mid)
    assert np.isclose(v_mid, 2.0, atol=1e-5)
    assert np.isclose(a_mid, 0.0, atol=1e-5)

    # Final state
    sf, vf, af = prof.evaluate(T)
    assert np.isclose(sf, qf, atol=1e-5)
    assert np.isclose(vf, 0.0, atol=1e-5)


def test_trapezoidal_profile_triangular():
    # Short move that cannot reach v_max
    prof = TrapezoidalVelocityProfile(max_velocity=10.0, max_acceleration=2.0)
    q0, qf = 0.0, 1.0
    ta, tc, td, T = prof.plan(q0, qf)

    assert np.isclose(tc, 0.0)
    sf, vf, _ = prof.evaluate(T)
    assert np.isclose(sf, qf, atol=1e-5)
    assert np.isclose(vf, 0.0, atol=1e-5)


def test_natural_cubic_spline_interpolation():
    t_pts = [0.0, 1.0, 2.5, 4.0]
    y_pts = [0.0, 2.0, 1.5, 5.0]

    spline = NaturalCubicSpline1D(t_pts, y_pts)

    # Verify interpolation at exact knots
    for t_i, y_i in zip(t_pts, y_pts):
        pos, _, _ = spline.evaluate(t_i)
        assert np.isclose(pos, y_i, atol=1e-6)

    # Contin