import numpy as np
import modeling as md
import modeling_casadi as mdca
import traj_opt

np.printoptions(precision=2)

N, M, K = traj_opt.N, traj_opt.M, traj_opt.K

sol = np.load("./traj_opt_npy/run_0010.npy")

traj_x = traj_opt.extract_traj(sol)
traj_u = traj_opt.extract_control(sol)

print(traj_x.shape)


Q = np.eye(N)
Rw = 0.1 * np.eye(M)
Qf = np.eye(N)


def riccati_lqr():
    Ps = [np.zeros((N, N))] * K
    Ks = [np.zeros((M, N))] * (K - 1)

    Ps[K - 1] = Qf
    for k in range(K - 2, -1, -1):
        xk = traj_x[:, k]
        uk = traj_u[:, k]

        Ak = mdca.f_J_x_rk4(xk, uk)
        Bk = mdca.f_J_u_rk4(xk, uk)

        tmp1 = Rw + Bk.T @ Ps[k + 1] @ Bk
        tmp2 = Bk.T @ Ps[k + 1] @ Ak
        Ks[k] = np.linalg.solve(tmp1, tmp2)
        tmp = Ak - Bk @ Ks[k]
        Ps[k] = Q + Ks[k].T @ Rw @ Ks[k] + tmp.T @ Ps[k + 1] @ tmp

    return Ks, Ps


def my_controller(x, Ks, k, max_u=np.array([20])):
    control = None

    xk = traj_x[:, k].reshape((N, 1))
    uk = traj_u[:, k].reshape((M, 1))

    u = uk - Ks[k] @ (x - xk)

    control = np.minimum(np.maximum(-max_u, u), max_u)

    return control


if __name__ == "__main__":
    x_init = np.array([[0.0, np.pi, np.pi, 0.0, 0.0, 0.0]]).T
    x_target = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    u = np.array([[0.0]])

    Ks, _ = riccati_lqr()

    # simulation
    T = 5
    t = 0
    dt = 0.05

    states = x_init
    controls = u

    x = np.copy(x_init)

    while t < T - md.DT:
        k = t / md.DT
        k = int(np.round(k))

        noise = np.random.normal(0, 1e-2, (N, 1))
        noise = 0
        u = my_controller(x + noise, k=k, Ks=Ks)

        x = md.cart_penulum_rk4(x, u, dt=dt)

        if k == 0:
            controls = u.copy()
            states = x_init
        else:
            states = np.hstack((states, x))
            controls = np.hstack((controls, u))

        t += dt

    md.visualize([traj_x, states], [traj_u, controls], ["trajectory_", "control_"])
    # md.plt.show()
    md.animate(states[:, ::1], show_trace=True, init_state=True)
