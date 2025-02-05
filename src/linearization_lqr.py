import modeling as md
import jax.numpy as jnp
from jax import jacfwd, jit
import numpy as np


# from jax import config
# config.update("jax_enable_x64", True)


def linearize(x_bar, u_bar):
    # auto diff
    A = jit(jacfwd(md.cart_penulum_rk4, 0))(x_bar, u_bar).reshape(6, 6)
    B = jit(jacfwd(md.cart_penulum_rk4, 1))(x_bar, u_bar).reshape(6, 1)

    return A, B


def infinite_lqr(A, B, Qn, Q, R, K=5000):

    # Init
    Ps = [np.zeros((6, 6))] * K
    Ks = [np.zeros((1, 6))] * (K - 1)

    # Riccati backward
    Ps[K - 1] = Qn
    for k in range(K - 2, -1, -1):
        tmp1 = R + B.T @ Ps[k + 1] @ B
        tmp2 = B.T @ Ps[k + 1] @ A
        Ks[k] = np.linalg.solve(tmp1, tmp2)
        ## From DP
        tmp = A - B @ Ks[k]
        Ps[k] = Q + Ks[k].T @ R @ Ks[k] + tmp.T @ Ps[k + 1] @ tmp
        ## End from DP

    # Let's take out the results we need
    Kinf = Ks[0]
    Pinf = Ps[0]

    # md.plt.plot(Ks)
    # md.plt.plot(Ps)
    # md.plt.show()

    return Pinf, Kinf


if __name__ == "__main__":

    x_bar = np.array([[0.0, np.pi, np.pi, 0.0, 0.0, 0.0]]).T
    u_bar = np.array([[0.0]]).T
    u_min = np.ones_like(u_bar) * (-20)
    u_max = np.ones_like(u_bar) * (20)

    Q = 1 * np.eye(6)
    R = 0.1 * np.eye(1)
    QN = np.eye(6)

    A, B = linearize(x_bar, u_bar)

    P, K = infinite_lqr(A, B, QN, Q, R)

    # print(linearize(x_bar, u_bar)[1])
    # print(P, K)

    # simulation
    T = 10
    t = 0

    # x = np.copy(x_bar)
    x = np.array([[0.0, np.pi, np.pi, 0.0, 0.0, 0.0]]).T
    states = x
    controls = np.copy(u_bar)

    x_ref = np.array([[0.0, np.pi, np.pi / 2, 0.0, 0.0, 0.0]]).T

    while t < T:

        u = u_bar - K @ (x - x_ref)

        u = np.maximum(np.minimum(u, u_max), u_min)

        x = md.cart_penulum_rk4(x, u)

        states = np.hstack((states, x))
        controls = np.hstack((controls, u))
        t += md.DT

    md.visualize(states)
    md.animate(states)
    md.plt.show()
