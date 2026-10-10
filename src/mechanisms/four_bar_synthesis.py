"""
Planar Four-Bar Mechanism Synthesis & Freudenstein Analysis
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def verify_grashof(r1: float, r2: float, r3: float, r4: float) -> Dict[str, object]:
    """
    Grashof Condition for Four-Bar Linkage:
        s + l <= p + q
    where s is shortest link, l is longest, and p, q are intermediate links.
    """
    lengths = sorted([r1, r2, r3, r4])
    s, p, q, l = lengths[0], lengths[1], lengths[2], lengths[3]

    is_grashof = (s + l) <= (p + q)

    if is_grashof:
        if s == r2:
            mech_type = "Crank-Rocker"
        elif s == r1:
            mech_type = "Double-Crank (Drag-Link)"
        elif s == r3:
            mech_type = "Double-Rocker"
        else:
            mech_type = "Rocker-Crank"
    else:
        mech_type = "Triple-Rocker (Non-Grashof)"

    return {
        "is_grashof": bool(is_grashof),
        "shortest": s,
        "longest": l,
        "s_plus_l": s + l,
        "p_plus_q": p + q,
        "type": mech_type
    }


def solve_freudenstein_displacement(r1: float, r2: float, r3: float, r4: float, theta2: float) -> Optional[Tuple[float, float]]:
    """
    Solves output rocker angle theta4 and coupler angle theta3 given input crank angle theta2
    using Freudenstein's algebraic loop-closure equation:
        K1 * cos(theta4) - K2 * cos(theta2) + K3 = cos(theta2 - theta4)
    where:
        K1 = r1 / r2
        K2 = r1 / r4
        K3 = (r1^2 + r2^2 - r3^2 + r4^2) / (2 * r2 * r4)
    """
    K1 = r1 / r2
    K2 = r1 / r4
    K3 = (r1**2 + r2**2 - r3**2 + r4**2) / (2.0 * r2 * r4)

    A = np.cos(theta2) - K1 - K2 * np.cos(theta2) + K3
    B = -2.0 * np.sin(theta2)
    C = K1 - (K2 + 1.0) * np.cos(theta2) + K3

    disc = B**2 - 4.0 * A * C
    if disc < 0.0:
        return None  # Linkage cannot close

    # Quadratic solution for t = tan(theta4 / 2)
    t = (-B - np.sqrt(disc)) / (2.0 * A)
    theta4 = 2.0 * np.arctan(t)

    # Solve for coupler angle theta3:
    # r2 * cos(t2) + r3 * cos(t3) = r1 + r4 * cos(t4)
    # r2 * sin(t2) + r3 * sin(t3) = r4 * sin(t4)
    num = r4 * np.sin(theta4) - r2 * np.sin(theta2)
    den = r1 + r4 * np.cos(theta4) - r2 * np.cos(theta2)
    theta3 = np.arctan2(num, den)

    return float(theta3), float(theta4)


def synthesize_three_precision_points(theta2_pts: List[float],
                                      theta4_pts: List[float],
                                      ground_length_r1: float = 1.0) -> Tuple[float, float, float, float]:
    """
    Synthesizes a 4-bar mechanism given 3 precision coordination point pairs (theta2_i, theta4_i).
    Solves the 3x3 Freudenstein linear system for [K1, K2, K3]^T:
        [cos(theta4_i)  -cos(theta2_i)  1] * [K1, K2, K3]^T = cos(theta2_i - theta4_i)
    Returns:
        (r1, r2, r3, r4)
    """
    if len(theta2_pts) != 3 or len(theta4_pts) != 3:
        raise ValueError("Exactly 3 precision points required for Freudenstein synthesis.")

    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)

    for i in range(3):
        th2, th4 = theta2_pts[i], theta4_pts[i]
        A[i] = [np.cos(th4), -np.cos(th2), 1.0]
        b[i] = np.cos(th2 - th4)

    K = np.linalg.solve(A, b)
    K1, K2, K3 = K[0], K[1], K[2]

    r1 = float(ground_length_r1)
    r2 = r1 / K1
    r4 = r1 / K2
    r3_sq = r1**2 + r2**2 + r4**2 - 2.0 * r2 * r4 * K3
    if r3_sq <= 0:
        raise ValueError("Synthesis yielded invalid non-real coupler length.")
    r3 = np.sqrt(r3_sq)

    return float(r1), float(r2), float(r3), float(r4)
