"""
Standard Discrete Linear Kalman Filter
Author: K.V. Sai Manohar Prasad <prasad.sai718@gmail.com>
License: Apache-2.0
"""

from typing import Optional, Tuple
import numpy as np


class LinearKalmanFilter:
    """
    Standard Discrete-Time Linear Kalman Filter:
        State Equation: x_k = F * x_(k-1) + B * u_k + w_k,  w_k ~ N(0, Q)
        Measurement Equation: z_k = H * x_k + v_k,            v_k ~ N(0, R)
    """
    def __init__(self,
                 F: np.ndarray,
                 H: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray,
                 B: Optional[np.ndarray] = None,
                 x0: Optional[np.ndarray] = None,
                 P0: Optional[np.ndarray] = None):
        self.F = np.asarray(F, dtype=np.float64)
        self.H = np.asarray(H, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        
        self.dim_x = self.F.shape[0]
        self.dim_z = self.H.shape[0]

        self.B = np.zeros((self.dim_x, 1)) if B is None else np.asarray(B, dtype=np.float64)
        self.x = np.zeros(self.dim_x) if x0 is None else np.asarray(x0, dtype=np.float64).flatten()
        self.P = np.eye(self.dim_x) * 1.0 if P0 is None else np.asarray(P0, dtype=np.float64)

    def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Time update (predict step):
            x = F * x + B * u
            P = F * P * F^T + Q
        """
        if u is not None and self.B is not None:
            self.x = np.dot(self.F, self.x) + np.dot(self.B, np.asarray(u, dtype=np.float64).flatten())
        else:
            self.x = np.dot(self.F, self.x)

        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Measurement update (correction step) using numerically stable Joseph form:
            y = z - H * x
            S = H