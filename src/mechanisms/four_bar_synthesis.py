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
    K1 