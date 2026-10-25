# Fundamental Robotics and Control Algorithms
### Open-Source Computational Framework for Kinematics, Modern Control, State Estimation, and Trajectory Synthesis

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Testing: Pytest 100% Pass](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)](tests/)
[![Mathematics: Rigorous SO(3) / DARE / EKF](https://img.shields.io/badge/Math-Rigorous%20Derivations-orange.svg)](src/)

Authored and maintained by **K.V. Sai Manohar Prasad** (<prasad.sai718@gmail.com>)  
*Senior Robotics & Control Systems Software Engineer*

---

## 1. Architectural Overview & System Taxonomy

This repository provides production-grade, mathematically rigorous Python implementations of core algorithms spanning modern robotics, feedback control theory, dynamic state estimation, and mechanical motion synthesis. All algorithms are implemented without heavy black-box dependencies, emphasizing clear state-space representations, numerical stability, and verifiable unit tests.

```
+---------------------------------------------------------------------------------------+
|                       ROBOTICS AND CONTROL ALGORITHMS FRAMEWORK                       |
+---------------------------------------------------------------------------------------+
|  KINEMATICS          |  CONTROL             |  ESTIMATION         |  TRAJECTORY       |
|  - SO(3)/SE(3) Trans |  - Discrete PID      |  - Linear Kalman    |  - Trapezoidal    |
|  - Rodrigues Axis    |  - DARE LQR Optimal  |  - Non-linear EKF   |  - Natural Spline |
|  - Planar 2R/3R Arms |  - Anti-Windup       |  - Complementary    |  - Minimum Jerk   |
|  - Standard D-H      |  - Feedforward Torq  |  - Joseph Covariance|  - 5th-Order Poly |
+---------------------------------------------------------------------------------------+
|                               MECHANISM SYNTHESIS                                     |
|  - Freudenstein Function Generation  |  - Grashof Rotatability  |  - Transmission Angle |
+---------------------------------------------------------------------------------------+
```

---

## 2. Core Modules & Mathematical Foundations

### 2.1 Spatial Kinematics (`src/kinematics/`)

* **Rodrigues' Rotation Formula:** Computes rotation matrix $\mathbf{R} \in \text{SO}(3)$ given unit axis $\mathbf{k}$ and angle $\theta$:
  $$\mathbf{R} = \mathbf{I} + \sin\theta [\mathbf{k}]_\times + (1 - \cos\theta) [\mathbf{k}]_\times^2$$
  where $[\mathbf{k}]_\times$ is the skew-symmetric cross-product matrix.
* **Unit Quaternions:** Compact parameterization $\mathbf{q} = [q_x, q_y, q_z, q_w]^T$ avoiding gimbal lock, with Hamilton product composition:
  $$\mathbf{q}_1 \otimes \mathbf{q}_2 = \begin{bmatrix} w_1 \mathbf{v}_2 + w_2 \mathbf{v}_1 + \mathbf{v}_1 \times \mathbf{v}_2 \\ w_1 w_2 - \mathbf{v}_1 \cdot \mathbf{v}_2 \end{bmatrix}$$
* **Planar Kinematics:** Analytical closed-form forward and inverse kinematics for 2-link and 3-link planar revolute manipulators with wrist-decoupled orientation.
* **Denavit-Hartenberg (D-H):** Standard homogeneous matrix construction chaining link offsets, twists, and revolute joint displacements:
  $$\mathbf{A}_i = \text{Rot}_z(\theta_i) \cdot \text{Trans}_z(d_i) \cdot \text{Trans}_x(a_i) \cdot \text{Rot}_x(\alpha_i)$$

---

### 2.2 Modern Control Theory (`src/control/`)

* **Discrete PID with Anti-Windup (`pid_controller.py`):**
  * Tustin trapezoidal numerical integration:
    $$I[k] = I[k-1] + K_i \frac{\Delta t}{2} (e[k] + e[k-1])$$
  * First-order low-pass derivative filtering with cutoff time constant $\tau$:
    $$D[k] = \frac{\tau}{\tau + \Delta t} D[k-1] + \frac{K_d}{\tau + \Delta t} (e[k] - e[k-1])$$
  * Dynamic conditional clamping preventing actuator saturation runaway.
* **Discrete Linear Quadratic Regulator (`lqr_regulator.py`):**
  * Solves the infinite-horizon Discrete Algebraic Riccati Equation (DARE):
    $$\mathbf{P} = \mathbf{A}^T \mathbf{P} \mathbf{A} - (\mathbf{A}^T \mathbf{P} \mathbf{B}) (\mathbf{R} + \mathbf{B}^T \mathbf{P} \mathbf{B})^{-1} (\mathbf{B}^T \mathbf{P} \mathbf{A}) + \mathbf{Q}$$
  * Generates optimal state-feedback gain matrix $\mathbf{K} = (\mathbf{R} + \mathbf{B}^T \mathbf{P} \mathbf{B})^{-1} \mathbf{B}^T \mathbf{P} \mathbf{A}$ guaranteeing asymptotic closed-loop stability ($\max |\lambda(\mathbf{A} - \mathbf{B}\mathbf{K})| < 1$).
* **Feedforward Torque Compensation (`feedforward.py`):**
  * Evaluates gravity loading $\mathbf{g}(\mathbf{q})$, viscous friction $\mathbf{F}_v \dot{\mathbf{q}}$, and Coulomb friction $\mathbf{F}_c \text{sgn}(\dot{\mathbf{q}})$ for dynamic tracking control.

---

### 2.3 Dynamic State Estimation (`src/estimation/`)

* **Discrete Linear Kalman Filter (`kalman_filter.py`):**
  * State prediction and measurement update with Joseph stabilized covariance formulation ensuring numerical positive-definiteness:
    $$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k \mathbf{R} \mathbf{K}_k^T$$
* **Extended Kalman Filter (`extended_kalman_filter.py`):**
  * Non-linear unicycle mobile robot motion model driven by linear and angular velocities $(v, \omega)$.
  * Analytical Jacobian propagation $\mathbf{F}_x = \left.\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right|_{\hat{\mathbf{x}}}$ and innovation angle wrapping.
* **Complementary Filter (`complementary_filter.py`):**
  * Fuses rate gyroscope integration with accelerometer gravity projections for low-latency attitude estimation:
    $$\theta_k = \alpha (\theta_{k-1} + \omega \Delta t) + (1 - \alpha) \theta_{acc}$$

---

### 2.4 Trajectory Synthesis (`src/trajectory/`)

* **Trapezoidal Velocity Profile (`trapezoidal_profile.py`):**
  * Bang-coast-bang symmetric acceleration planning with automatic triangular profile detection when target displacement is insufficient to reach peak velocity $v_{max}$.
* **Natural Cubic Spline (`cubic_spline.py`):**
  * $C^2$ continuous piecewise cubic interpolation through arbitrary waypoint knots with tridiagonal curvature solver.
* **Minimum Jerk Trajectory (`minimum_jerk.py`):**
  * Flash & Hogan (1985) model minimizing normalized jerk integral $\int_0^T (\dddot{x}(t))^2 dt$:
    $$x(t) = x_0 + (x_f - x_0) (10 \tau^3 - 15 \tau^4 + 6 \tau^5), \quad \tau = \frac{t}{T}$$

---

### 2.5 Mechanism Synthesis (`src/mechanisms/`)

* **Grashof Condition (`four_bar_synthesis.py`):**
  * Evaluates mobility inequality $s + l \le p + q$ to classify planar linkages into Crank-Rocker, Double-Crank, Double-Rocker, or Non-Grashof mechanisms.
* **Freudenstein Function Synthesis:**
  * Coordinates input crank $\theta_2$ and output rocker $\theta_4$ across 3 precision points by solving:
    $$K_1 \cos\theta_4 - K_2 \cos\theta_2 + K_3 = \cos(\theta_2 - \theta_4)$$
* **Transmission Angle Analysis (`transmission_angle.py`):**
  * Analyzes force transmission angle $\mu$ over $360^\circ$ crank rotation, verifying industrial compliance within the permissible zone $45^\circ \le \mu \le 135^\circ$.

---

## 3. Repository Structure

```
robotics-and-control-algorithms/
├── src/
│   ├── kinematics/
│   │   ├── transformations.py        # SO(3)/SE(3), Rodrigues, Quaternions
│   │   ├── planar_kinematics.py      # Analytical 2R and 3R arm solvers
│   │   └── dh_matrix.py              # Denavit-Hartenberg transformation matrix builder
│   ├── control/
│   │   ├── pid_controller.py         # Discrete PID with anti-windup clamping
│   │   ├── lqr_regulator.py          # Discrete Algebraic Riccati LQR
│   │   └── feedforward.py            # Gravity and friction torque compensation
│   ├── estimation/
│   │   ├── kalman_filter.py          # Linear Kalman Filter with Joseph form
│   │   ├── extended_kalman_filter.py # EKF for non-linear robot pose fusion
│   │   └── complementary_filter.py   # IMU accelerometer + gyro attitude fusion
│   ├── trajectory/
│   │   ├── trapezoidal_profile.py    # Symmetric bang-coast-bang profile
│   │   ├── cubic_spline.py           # C2 continuous natural cubic spline
│   │   └── minimum_jerk.py           # 5th-order minimum jerk trajectory
│   └── mechanisms/
│       ├── four_bar_synthesis.py     # Freudenstein 3-point synthesis & Grashof
│       └── transmission_angle.py     # Transmission angle optimizer
├── tests/
│   ├── test_transformations.py
│   ├── test_control.py
│   ├── test_estimation.py
│   ├── test_trajectory.py
│   └── test_mechanisms.py
├── requirements.txt
└── README.md
```

---

## 4. Verification & Unit Testing

The repository includes a comprehensive unit testing suite using `pytest`.

```bash
# Clone the repository
git clone https://github.com/saimanohar-kv/robotics-and-control-algorithms.git
cd robotics-and-control-algorithms

# Install dependencies
pip install -r requirements.txt

# Run full test suite
pytest -v tests/
```

### Verified Test Results
* `test_transformations.py` -> **PASSED** (Orthonormality, Euler round-trip, Rodrigues & quaternion equivalence)
* `test_control.py` -> **PASSED** (Anti-windup clamping, linear plant tracking, LQR Riccati stability)
* `test_estimation.py` -> **PASSED** (Linear KF tracking, EKF non-linear covariance convergence, IMU attitude fusion)
* `test_trajectory.py` -> **PASSED** (Trapezoidal profile bounds, spline knot continuity, minimum jerk zero boundary jerk)
* `test_mechanisms.py` -> **PASSED** (Grashof classification, Freudenstein 3-point synthesis, transmission angle envelope)

---

## 5. License

Distributed under the **Apache-2.0 License**. See `LICENSE` for details.
