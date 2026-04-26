import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.linalg import solve_continuous_are
from scipy.signal import cont2discrete

# ==========================================================
# Inverted Pendulum on Cart
# Professional Academic Version
# Linear LQR + Linear MPC + Nonlinear Plant Simulation
# ==========================================================

# -----------------------------
# Physical Parameters
# -----------------------------
M = 1.0       # cart mass (kg)
m = 0.1       # pendulum mass (kg)
b = 0.1       # cart friction coefficient
l = 0.5       # pendulum COM length (m)
I = 0.006     # pendulum inertia
g = 9.81      # gravity (m/s^2)

# Force constraints
U_MIN = -20.0
U_MAX = 20.0


# ==========================================================
# NONLINEAR DYNAMICS
# state = [x, x_dot, theta, theta_dot]
# theta = 0 means upright
# ==========================================================
def pendulum_dynamics(state, t, F):
    x, x_dot, theta, theta_dot = state

    p = I * (M + m) + M * m * l**2

    x_ddot = (
        (I + m * l**2) * (F - b * x_dot)
        + m**2 * l**2 * g * np.sin(theta) * np.cos(theta)
        - (I + m * l**2) * m * l * theta_dot**2 * np.sin(theta)
    ) / p

    theta_ddot = (
        m * l * np.cos(theta) * (F - b * x_dot)
        + (M + m) * m * g * l * np.sin(theta)
        - m**2 * l**2 * theta_dot**2 * np.sin(theta) * np.cos(theta)
    ) / p

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


# ==========================================================
# LINEARIZATION AROUND UPRIGHT EQUILIBRIUM
# ==========================================================
def linearize():
    p = I * (M + m) + M * m * l**2

    A = np.array([
        [0, 1, 0, 0],
        [0, -(I + m * l**2) * b / p, m**2 * g * l**2 / p, 0],
        [0, 0, 0, 1],
        [0, -m * l * b / p, m * g * l * (M + m) / p, 0]
    ])

    B = np.array([
        [0],
        [(I + m * l**2) / p],
        [0],
        [m * l / p]
    ])

    return A, B


# ==========================================================
# DISCRETIZATION
# ==========================================================
def discretize(dt):
    A, B = linearize()

    C = np.zeros((4, 4))
    D = np.zeros((4, 1))

    Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), dt)

    return Ad, Bd.flatten()


# ==========================================================
# LQR CONTROLLER
# ==========================================================
def lqr_gain(Q, R):
    A, B = linearize()
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K.flatten()


# ==========================================================
# SIMULATE LQR ON NONLINEAR SYSTEM
# ==========================================================
def simulate_lqr(K, t_span, x0, disturbance_t=None, disturbance_mag=0):
    states = [np.array(x0, dtype=float)]
    forces = []

    dt = t_span[1] - t_span[0]
    x = np.array(x0, dtype=float)

    for t in t_span[:-1]:
        F = -K @ x
        F = np.clip(F, U_MIN, U_MAX)

        if disturbance_t is not None and abs(t - disturbance_t) < dt:
            F += disturbance_mag

        forces.append(F)

        x = odeint(
            lambda s, tau: pendulum_dynamics(s, tau, F),
            x,
            [t, t + dt]
        )[-1]

        states.append(x)

    forces.append(0.0)

    return np.array(states), np.array(forces)


# ==========================================================
# MPC CONTROLLER
# ==========================================================
def mpc_control(x_current, N=10, dt=0.05, u_prev=None):
    Ad, Bd = discretize(dt)

    Q = np.diag([2, 0.5, 15, 1])
    R = 0.1

    x_current = np.array(x_current, dtype=float).flatten()

    if u_prev is None:
        u0 = np.zeros(N)
    else:
        u0 = np.append(u_prev[1:], u_prev[-1])

    def cost(u_seq):
        x = x_current.copy()
        J = 0.0

        for k in range(N):
            u = u_seq[k]
            x = Ad @ x + Bd * u

            J += x @ Q @ x
            J += R * u**2

        return J

    bounds = [(U_MIN, U_MAX)] * N

    res = minimize(
        cost,
        u0,
        method="SLSQP",
        bounds=bounds,
        options={
            "maxiter": 40,
            "ftol": 1e-4,
            "disp": False
        }
    )

    if res.success:
        return res.x[0], res.x
    else:
        return 0.0, np.zeros(N)


# ==========================================================
# SIMULATE MPC ON NONLINEAR SYSTEM
# ==========================================================
def simulate_mpc(t_span, x0, N=10, disturbance_t=None, disturbance_mag=0):
    states = [np.array(x0, dtype=float)]
    forces = []

    dt = t_span[1] - t_span[0]
    x = np.array(x0, dtype=float)

    warm_start = None

    for t in t_span[:-1]:
        F, warm_start = mpc_control(
            x,
            N=N,
            dt=dt,
            u_prev=warm_start
        )

        if disturbance_t is not None and abs(t - disturbance_t) < dt:
            F += disturbance_mag

        F = np.clip(F, U_MIN, U_MAX)

        forces.append(F)

        x = odeint(
            lambda s, tau: pendulum_dynamics(s, tau, F),
            x,
            [t, t + dt]
        )[-1]

        states.append(x)

    forces.append(0.0)

    return np.array(states), np.array(forces)


# ==========================================================
# QUICK TEST
# ==========================================================
if __name__ == "__main__":
    t = np.linspace(0, 5, 200)
    x0 = [0, 0, np.radians(12), 0]

    Q = np.diag([1, 1, 10, 1])
    R = np.array([[0.1]])

    K = lqr_gain(Q, R)

    sol_lqr, F_lqr = simulate_lqr(K, t, x0)
    sol_mpc, F_mpc = simulate_mpc(t, x0, N=10)

    print("LQR Final Angle (deg):", np.degrees(sol_lqr[-1, 2]))
    print("MPC Final Angle (deg):", np.degrees(sol_mpc[-1, 2]))