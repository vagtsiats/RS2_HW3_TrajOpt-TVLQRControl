import modeling as md
import modeling_casadi as mdca
import numpy as np
import cyipopt
import os

N = 6  # state size
M = 1  # control size

tf = 5
K = round(tf / md.DT) + 1  # node points

x_dim = K * N  # states
x_dim += (K - 1) * M  # controls
eq_dim = (K - 1) * N  # dynamics
eq_dim += 2 * N  # initial and final state


def extract_traj(x_ipopt):
    states = x_ipopt[:N].reshape((N, 1))

    for k in range(1, K - 1, 1):
        states = np.hstack((states, x_ipopt[k * N : (k + 1) * N].reshape((N, 1))))

    # states = np.hstack((states, x_ipopt[k * N : (k + 1) * N].reshape((N, 1))))

    return states


def extract_control(x_ipopt):
    idx = K * N
    controls = x_ipopt[idx : idx + M].reshape((M, 1))

    for k in range(1, K - 2, 1):
        controls = np.hstack((controls, x_ipopt[idx + k * M : idx + (k + 1) * M].reshape((M, 1))))

    controls = np.hstack((controls, x_ipopt[idx + k * M : idx + (k + 1) * M].reshape((M, 1))))

    return controls


Q = 10 * np.eye(N)
R = 0.01 * np.eye(M)


class TrajOpt:
    def __init__(self, targ):
        self.target = targ
        pass

    def objective(self, x):
        obj = 0

        for k in range(K - 1):
            cart_x = x[k * N]
            state = x[k * N : (k + 1) * N]
            u = x[K * N + k * M : K * N + (k + 1) * M]

            # obj += cart_x**2
            # obj += u.T @ R @ u
            # obj += (state - self.target).T @ Q @ (state - self.target)

        return obj

    def gradient(self, x):
        obj_grad = np.zeros((x_dim,))

        for k in range(K - 1):
            state = x[k * N : (k + 1) * N]
            u = x[K * N + k * M : K * N + (k + 1) * M]
            cart_x = x[k * N]

            # obj_grad[k * N] = 2 * cart_x
            # obj_grad[k * N : (k + 1) * N] += 2 * (state - self.target).T @ Q
            # obj_grad[K * N + k * M : K * N + (k + 1) * M] = 2 * u.T @ R

        return obj_grad

    def constraints(self, x):
        c = np.zeros((eq_dim,))

        # dynamics constraints
        for k in range(K - 2):
            x0 = x[k * N : (k + 1) * N]
            x1 = x[(k + 1) * N : (k + 2) * N]
            u0 = x[K * N + k * M : K * N + (k + 1) * M]
            u1 = x[K * N + (k + 1) * M : K * N + (k + 2) * M]

            x_new = x0 + (md.pendulum(x0, u0).reshape((N,))) * md.DT
            x_new = x0 + 0.5 * (md.pendulum(x1, u1).reshape((N,)) + md.pendulum(x0, u0).reshape((N,))) * md.DT

            c[k * N : (k + 1) * N] = x1 - x_new

        ## Trapeoidal constraints
        k = K - 2
        x0 = x[k * N : (k + 1) * N]
        x1 = x[(k + 1) * N : (k + 2) * N]
        u0 = x[K * N + k * M : K * N + (k + 1) * M]

        c[k * N : (k + 1) * N] = x1 - (x0 + md.pendulum(x0, u0).reshape((N,)) * md.DT)

        x0 = x[:N]
        xN = x[(K - 1) * N : K * N]

        # Initial - final state constraints
        si = eq_dim - 2 * N
        c[si : si + N] = x0
        c[si + N :] = xN

        return c

    # TODO - sparse jacobian structure
    def jacobian(self, x):
        J = np.zeros((eq_dim, x_dim))
        for k in range(K - 2):
            x0 = x[k * N : (k + 1) * N]
            x1 = x[(k + 1) * N : (k + 2) * N]
            u0 = x[K * N + k * M : K * N + (k + 1) * M]
            u1 = x[K * N + (k + 1) * M : K * N + (k + 2) * M]

            dX0 = mdca.f_J_x(x0, u0)
            dU0 = mdca.f_J_u(x0, u0)
            dX1 = mdca.f_J_x(x1, u1)
            dU1 = mdca.f_J_u(x1, u1)

            # wrt x0
            J[k * N : (k + 1) * N, k * N : (k + 1) * N] = -np.eye(N) - 0.5 * dX0 * md.DT
            # wrt x1
            J[k * N : (k + 1) * N, (k + 1) * N : (k + 2) * N] = np.eye(N) - 0.5 * dX1 * md.DT
            # wrt u
            J[k * N : (k + 1) * N, K * N + k * M : K * N + (k + 1) * M] = -0.5 * (dU1 + dU0) * md.DT

        k = K - 2
        x0 = x[k * N : (k + 1) * N]
        x1 = x[(k + 1) * N : (k + 2) * N]
        u0 = x[K * N + k * M : K * N + (k + 1) * M]

        dX0 = mdca.f_J_x(x0, u0)
        dU0 = mdca.f_J_u(x0, u0)

        # wrt x0
        J[k * N : (k + 1) * N, k * N : (k + 1) * N] = -np.eye(N) - dX0 * md.DT
        # wrt x1
        J[k * N : (k + 1) * N, (k + 1) * N : (k + 2) * N] = np.eye(N)
        # wrt u
        J[k * N : (k + 1) * N, K * N + k * M : K * N + (k + 1) * M] = -dU0 * md.DT

        # Initial - final state constraints
        si = eq_dim - 2 * N
        J[si : si + N, :N] = np.eye(N)
        J[si + N :, (K - 1) * N : K * N] = np.eye(N)

        return J

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        # Example for the use of the intermediate callback.
        # print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
        pass


if __name__ == "__main__":
    x_init = np.array([[0.0, np.pi, np.pi, 0.0, 0.0, 0.0]]).T
    x_target = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    u = np.array([0.0])

    x0 = np.zeros((x_dim,))

    # Initialization
    for k in range(K):
        # x0[k * N + 1] = x_init[1] + (x_target - x_init)[1] * float(k) / (K - 1.0)
        # x0[k * N + 2] = x_init[2] + (x_target - x_init)[2] * float(k) / (K - 1.0)
        x0[k * N : (k + 1) * N] = x_init.reshape((-1,))

    lb = [None] * x_dim
    ub = [None] * x_dim

    # Add u bounds
    sidx = K * N
    for k in range(K - 1):
        lb[sidx + k * M : sidx + (k + 1) * M] = [-20] * M
        ub[sidx + k * M : sidx + (k + 1) * M] = [20] * M

    # print(lb)

    cl = [0.0] * eq_dim
    cu = [0.0] * eq_dim

    si = eq_dim - 2 * N
    # initial state
    for i in range(N):
        cl[si + i] = float(x_init[i, 0])
        cu[si + i] = float(x_init[i, 0])

    # final state
    for i in range(N):
        cl[si + N + i] = float(x_target[i, 0])
        cu[si + N + i] = float(x_target[i, 0])

    nlp = cyipopt.Problem(
        n=len(x0), m=len(cl), problem_obj=TrajOpt(x_target.reshape((-1,))), lb=lb, ub=ub, cl=cl, cu=cu
    )

    nlp.add_option("jacobian_approximation", "exact")
    nlp.add_option("print_level", 5)
    nlp.add_option("nlp_scaling_method", "none")
    # nlp.add_option("max_iter", 100)
    # nlp.add_option("tol", 1e-15)

    # Solve the problem
    x, info = nlp.solve(x0)

    run_id = 1
    while os.path.exists(f"traj_opt_npy/run_{run_id:04d}.npy"):
        run_id += 1

    # Save the array
    filename = f"traj_opt_npy/run_{run_id:04d}.npy"

    # Save the array
    np.save(filename, x)

    traj_states = extract_traj(x)
    traj_controls = extract_control(x)
    # print(traj_states)

    md.visualize(traj_states, controls=traj_controls)
    md.animate(traj_states, ref_state=x_target, init_state=True, show_trace=True)
    md.plt.show()
