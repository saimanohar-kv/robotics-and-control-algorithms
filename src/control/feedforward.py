"""
Gravity, Viscous, and Inertial Feedforward Torque Compensators
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import Optional
import numpy as np


class FeedforwardTorqueCompensator:
    """
    Computes feedforward dynamics compensation for robotic manipulators:
        tau_ff = M(q) * q_ddot_d + C(q, q_dot) * q_dot_d + g(q) + F_v * q_dot_d + F_c * sgn(q_dot_d)
    """
    def __init__(self,
                 masses: np.ndarray,
                 link_lengths: np.ndarray,
                 viscous_friction: Optional[np.ndarray] = None,
                 coulomb_friction: Optional[np.ndarray] = None,
                 gravity_accel: float = 9.80665):
        self.masses = np.asarray(masses, dtype=np.float64)
        self.link_lengths = np.asarray(link_lengths, dtype=np.float64)
        self.num_joints = len(self.masses)
        self.g = float(gravity_accel)

        self.f_v = np.zeros(self.num_joints) if viscous_friction is None else np.asarray(viscous_friction, dtype=np.float64)
        self.f_c = np.zeros(self.num_joints) if coulomb_friction is None else np.asarray(coulomb_friction, dtype=np.float64)

    def compute_gravity_compensation(self, q: np.ndarray) -> np.ndarray:
        """
        Analytical gravity torque compensation vector g(q) for planar serial arm.
        Assuming point masses at the distal ends of each link.
        """
        tau_g = np.zeros(self.num_joints, dtype=np.float64)
        cumulative_angle = 0.0

        for i in range(self.num_joints):
            cumulative_angle += q[i]
            # Lever arm contribution of remaining downstream masses
            downstream_mass = np.sum(self.masses[i:])
            tau_g[i] = downstream_mass * self.g * self.link_lengths[i] * np.cos(cumulative_angle)

        return tau_g

    def compute_friction_compensation(self, q_dot_desired: np.ndarray) -> np.ndarray:
        """Computes viscous and Coulomb friction feedforward compensation."""
        qd = np.asarray(q_dot_desired, dtype=np.float64)
        tau_viscous = self.f_v * qd
        tau_coulomb = self.f_c * np.sign(qd)
        return tau_viscous + tau_coulomb

    def compute_feedforward(self,
                            q_desired: np.ndarray,
                            q_dot_desired: np.ndarray,
                            q_ddot_desired: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Full feedforward control torque calculation: Gravity + Friction + Inertia.
        """
        tau_gravity = self.compute_gravity_compensation(q_desired)
        tau_friction = self.compute_friction_compensation(q_dot_desired)
        
        tau_inertia = np.zeros(self.num_joints)
        if q_ddot_desired is not None:
            # Approximate diagonal joint inertia
            I_diag = self.masses * (self.link_lengths ** 2)
            tau_inertia = I_diag * np.asarray(q_ddot_desired, dtype=np.float64)

        return tau_gravity + tau_friction + tau_inertia
