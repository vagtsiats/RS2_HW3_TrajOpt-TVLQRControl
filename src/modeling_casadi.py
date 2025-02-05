import casadi as ca

import modeling as md
import numpy as np
import time


m1, m2, m3 = 0.5, 0.5, 0.5
l2, l3 = 0.6, 0.6
b = 0.1
g = 9.81
i2 = (1 / 12) * m2 * l2**2
i3 = (1 / 12) * m3 * l3**2


def ca_pendulum(x, u):
    x = ca.reshape(x, 6, 1)
    u = ca.reshape(u, 1, 1)[0, 0]

    # State variables
    x1 = x[0, 0]
    th2 = x[1, 0]
    th3 = x[2, 0]
    x1_dot = x[3, 0]
    th2_dot = x[4, 0]
    th3_dot = x[5, 0]

    # Define alphas
    alphas = ca.vertcat(
        u - b * x1_dot - 0.5 * (m2 + 2 * m3) * l2 * th2_dot**2 * ca.sin(th2) - 0.5 * m3 * l3 * th3_dot**2 * ca.sin(th3),
        (0.5 * m2 + m3) * l2 * g * ca.sin(th2) - 0.5 * m3 * l2 * l3 * th3_dot**2 * ca.sin(th2 - th3),
        0.5 * m3 * l3 * (g * ca.sin(th3) + l2 * th2_dot**2 * ca.sin(th2 - th3)),
    )

    # Define matrix mat
    mat = ca.vertcat(
        ca.horzcat(m1 + m2 + m3, -(0.5 * m2 + m3) * l2 * ca.cos(th2), -0.5 * m3 * l3 * ca.cos(th3)),
        ca.horzcat(
            -(0.5 * m2 + m3) * l2 * ca.cos(th2),
            m3 * l2**2 + i2 + 0.25 * m2 * l2**2,
            0.5 * m3 * l2 * l3 * ca.cos(th2 - th3),
        ),
        ca.horzcat(-0.5 * m3 * l3 * ca.cos(th3), 0.5 * m3 * l2 * l3 * ca.cos(th2 - th3), 0.25 * m2 * l3**2 + i3),
    )

    # Solve for accelerations
    ddot = ca.mtimes(ca.inv(mat), alphas)

    # Compute x_dot
    x_dot = ca.vertcat(x[3:, :], ddot)

    return x_dot


def rk4_step(f, x, u, dt=md.DT):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


x = ca.MX.sym("x", 6)
u = ca.MX.sym("u", 1)


x_dot = ca_pendulum(x, u)
x_rk4 = rk4_step(ca_pendulum, x, u)

J_x = ca.jacobian(x_dot, x)
J_u = ca.jacobian(x_dot, u)

J_x_rk4 = ca.jacobian(x_rk4, x)
J_u_rk4 = ca.jacobian(x_rk4, u)

# Create CasADi functions for evaluation
f_x_dot = ca.Function("x_dot", [x, u], [x_dot])
f_x_rk4 = ca.Function("x_rk4", [x, u], [x_rk4])
f_J_x = ca.Function("J_x", [x, u], [J_x])
f_J_u = ca.Function("J_u", [x, u], [J_u])
f_J_x_rk4 = ca.Function("J_x_rk4", [x, u], [J_x_rk4])
f_J_u_rk4 = ca.Function("J_u_rk4", [x, u], [J_u_rk4])


if __name__ == "__main__":

    # Evaluate with specific values
    x_val = np.array([[0.0, 0.1, 0.2, 1, 0, 0]]).T
    u_val = np.array([10.0])

    jx_val = md.jnp.array(x_val)
    ju_val = md.jnp.array(u_val)

    t0 = time.time()
    J_x_ca = f_J_x(x_val, u_val)
    t1 = time.time()
    J_u_ca = f_J_u(x_val, u_val)
    t2 = time.time()
    J_x_ja = md.pend_derivX(jx_val, ju_val)
    t3 = time.time()
    J_u_ja = md.pend_derivU(x_val, u_val)
    t4 = time.time()

    print(np.linalg.norm(np.array(J_x_ca) - J_x_ja))
    print(t1 - t0, t3 - t2)
