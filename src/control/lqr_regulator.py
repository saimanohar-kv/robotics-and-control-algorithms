"""
Discrete-Time Linear Quadratic Regulator (LQR) using Algebraic Riccati Equation
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import Tuple
import numpy as np
from scipy import linalg


class DiscreteLQRRegulator:
    """
    Solves infinite-horizon discrete-time LQR optimal state feedback control:
        System: x[k+1] = A x[k] + B u[k]
        Cost: J = sum(x[k]^T Q x[k] + u[k]^T R u[k])
        Feedback Law: u[k] = -K x[k]
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)

        self.P, self.K = self._solve_dare()

    def _solve_dare(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves Discrete Algebraic Riccati Equation (DARE):
            P = A^T P A - (A^T P B)(R + B^T P B)^(-1)(B^T P A) + Q
        """
        try:
            # Solves X = A^T X A - (A^T X B) (R + B^T X B)^(-1) (B^T X A) + Q
            P = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        except Exception:
            # Iterative fallback solver
            P = self.Q.copy()
            for _ in range(500):
                P_next = self.A.T @ P @ self.A - (self.A.T @ P @ self.B) @ \
                         linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A) + self.Q
                if np.allclose(P, P_next, atol=1e-8):
                    P = P_next
              