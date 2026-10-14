"""
Unit Tests for Mechanisms: Freudenstein Synthesis & Transmission Angle
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
"""

import numpy as np
import pytest
from src.mechanisms.four_bar_synthesis import (
    verify_grashof, solve_freudenstein_displacement, synthesize_three_precision_points
)
from src.mechanisms.transmission_angle import (
    compute_transmission_angle, evaluate_transmission_angle_range
)


def test_grashof_criterion():
    # Crank-Rocker: shortest link is input crank r2
    res = verify_grashof(r1=300.0, r2=75.0, r3=350.0, r4=250.0)
    assert res["is_grashof"] is True
    assert res["type"] == "Crank-Rocker"

    # Non-Grashof: sum of shortest and longest exceeds intermediate sum
    res_non = verify_grashof(r1=100.0, r2=120.0, r3=400.0, r4=150.0)
    assert res_non["is_grashof"] is False
    assert res_non["type"] == "Triple-Rocker (Non-Grashof)"


def test_freudenstein_loop_closure():
    r1, r2, r3, r4 = 300.0, 75.0, 350.0, 250.0
    theta2 = np.radians(45.0)

    sol = solve_freudenstein_displacement(r1, r2, r3, r4, theta2)
    assert sol is not None
    th3, th4 = sol

    # Vector loop closure residual check:
    # r2 * cos(t2) + r3 * cos(t3) - r4 * cos(t4) - r1 == 0
    # r2 * sin(t2) + r3 * sin(t3) - r4 * sin(t4) == 0
    res_x = r2 * np.cos(theta2) + r3 * np.cos(th3) - r4 * np.cos(th4) - r1
    res_y = r2 * np.sin(theta2) + r3 * np.sin(th3) - r4 * np.sin(th4)

    assert np.isclose(res_x, 0.0, atol=1e-5)
    assert np.isclose(res_y, 0.0, atol=1e-5)


def test_freudenstein_three_point_synthesis():
    # True mechanism
    r1_true, r2_true, r3_true, r4_true = 1.0, 0.3, 0.9, 0.7
    theta2_pts = [np.radians(30.0), np.radians(60.0), np.radians(90.0)]
    theta4_pts = []

    for th2 in theta2_pts:
        _, th4 = solve_freudenstein_displacement(r1_true, r2_true, r3_true, r4_true, th2)
        theta4_pts.append(th4)

    # Synthesize from precision points
    r1, r2, r3, r4 = synthesize_three_precision_points(theta2_pts, theta4_pts, ground_length_r1=1.0)

    assert np.isclose(r1, r1_true, atol=1e-4)
    assert np.isclose(r2, r2_true, atol=1e-4)
    assert np.isclose(r3, r3_true, atol=1e-4)
    assert np.isclose(r4, r4_true, atol=1e-4)


def test_transmission_angle_bounds():
    # Test well-proportioned 4-bar linkage
    r1, r2, r3, r4 = 1.0, 0.3, 0.9, 0.7
    eval_res = evaluate_transmission_angle_range(r1, r2, r3, r4)

    assert eval_res["is_permissible"] is True
    assert eval_res["min_mu_deg"] >= 45.0
    assert eval_res["max_mu_deg"] <= 135.0
