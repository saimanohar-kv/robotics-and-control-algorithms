"""
Transmission Angle Analysis & Force Transmission Optimization for Four-Bar Mechanisms
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import Dict, Tuple
import numpy as np


def compute_transmission_angle(r1: float, r2: float, r3: float, r4: float, theta2: float) -> float:
    """
    Computes transmission angle mu (angle between coupler link r3 and output rocker r4):
        cos(mu) = (r3^2 + r4^2 - r1^2 - r2^2 + 2 * r1 * r2 * cos(theta2)) / (2 * r3 * r4)
    Returns:
        mu in radians in [0, pi]
    """
    num = (r3**2 + r4**2 - r1**2 - r2**2 + 2.0 * r1 * r2 * np.cos(theta2))
    den = 2.0 * r3 * r4
    ratio = np.clip(num / den, -1.0, 1.0)
    return float(np.arccos(ratio))


def evaluate_transmission_angle_range(r1: float,
                                      r2: float,
                                      r3: float,
                                      r4: float,
                                      num_samples: int = 360) -> Dict[str, float]:
    """
    Evaluates minimum, maximum, and critical deviation of transmission angle over full 360 crank revolution.
    Industrial standard: Transmission angle mu should satisfy:
        45 deg <= mu <= 135 deg (|mu - 90 deg| <= 45 deg).
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)
    mu_values = np.array([compute_transmission_angle(r1, r2, r3, r4, th) for th in angles])

    min_mu_deg = float(np.degrees(np.min(mu_values)))
    max_mu_deg = float(np.degrees(np.max(mu_values)))
    
    # Critical deviation from ideal 90 deg
    dev_min = abs(90.0 - min_mu_deg)
    dev_max = abs(max_mu_deg - 90.0)
    max_deviation = max(dev_min, dev_max)

    is_optimal = (min_mu_deg >= 45.0) and (max_mu_deg <= 135.0)

    return {
        "min_mu_deg": min_mu_deg,
        "max_mu_deg": max_mu_deg,
        "max_deviation_from_90": max_deviation,
        "is_permissible": is_optimal
    }
