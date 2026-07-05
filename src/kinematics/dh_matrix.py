"""
Standard Denavit-Hartenberg Transformation Matrix Builder
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import List, Sequence
import numpy as np


def dh_standard_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Constructs 4x4 homogeneous transformation matrix using standard D-H convention:
        A_i = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0]
    ], dtype=np.float64)


def dh_chain(dh_table: Sequence[Sequence[float]], joint_variables: Sequence[float]) -> List[np.ndarray]:
    """
    Chains sequential D-H transformations.
    dh_table: List of tuples/lists each containing [d, a, alpha, theta_offset]
    joint_variables: List of joint angles [theta_1, theta_2, ..., theta_n]
    Returns: List of transforms [T_0_0, T_0_1, ..., T_0_n]
    """
    if len(dh_table) != len(joint_variables):
        raise ValueError("Number of D-H rows must match number of joint variables.")

    T = np.eye(4, dtype=np.float64)
    transforms = [T.copy()]

    for row, q in zip(dh_table, joint_variables):
        d, a, alpha = row[0], row[1], row[2]
        theta_offset = row[3] if len(row) > 3 else 0.0
        A = dh_standard_matrix(q + theta_offset, d, a, alpha)
        T = np.dot(T, A)
        transforms.append(T.copy())

    return transforms
