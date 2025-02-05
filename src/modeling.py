import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scienceplots
import time
import jax.numpy as jnp
from jax import jacfwd, jit

# import modeling_casadi as mdca

plt.style.use(["science", "no-latex", "nature", "grid"])
plt.rcParams.update({"figure.dpi": "300"})

m1, m2, m3 = 0.5, 0.5, 0.5
l2, l3 = 0.6, 0.6
b = 0.1
g = 9.81
i2 = (1 / 12) * m2 * l2**2
i3 = (1 / 12) * m3 * l3**2
DT = 0.05


def jpendulum(x, u):
    x = x.reshape((6, 1))
    u = u.reshape((1,))
    u = u[0]
    x1 = x[0, 0]
    th2 = x[1, 0]
    th3 = x[2, 0]
    x1_dot = x[3, 0]
    th2_dot = x[4, 0]
    th3_dot = x[5, 0]

    alphas = jnp.array(
        [
            [
                u
                - b * x1_dot
                - 0.5 * (m2 + 2 * m3) * l2 * th2_dot**2 * jnp.sin(th2)
                - 0.5 * m3 * l3 * th3_dot**2 * jnp.sin(th3),
                (0.5 * m2 + m3) * l2 * g * jnp.sin(th2) - 0.5 * m3 * l2 * l3 * th3_dot**2 * jnp.sin(th2 - th3),
                0.5 * m3 * l3 * (g * jnp.sin(th3) + l2 * th2_dot**2 * jnp.sin(th2 - th3)),
            ]
        ]
    ).T

    mat = jnp.array(
        [
            [m1 + m2 + m3, -(0.5 * m2 + m3) * l2 * jnp.cos(th2), -0.5 * m3 * l3 * jnp.cos(th3)],
            [
                -(0.5 * m2 + m3) * l2 * jnp.cos(th2),
                m3 * l2**2 + i2 + 0.25 * m2 * l2**2,
                0.5 * m3 * l2 * l3 * jnp.cos(th2 - th3),
            ],
            [-0.5 * m3 * l3 * jnp.cos(th3), 0.5 * m3 * l2 * l3 * jnp.cos(th2 - th3), 0.25 * m2 * l3**2 + i3],
        ]
    )

    ddot = jnp.linalg.solve(mat, alphas)

    x_dot = jnp.vstack((x[3:, :], ddot))

    return x_dot


def pendulum(x, u):
    x = x.reshape((6, 1))
    u = u.reshape((1,))
    u = u[0]
    x1 = x[0, 0]
    th2 = x[1, 0]
    th3 = x[2, 0]
    x1_dot = x[3, 0]
    th2_dot = x[4, 0]
    th3_dot = x[5, 0]

    alphas = np.zeros((3, 1))
    alphas[0, 0] = (
        u - b * x1_dot - 0.5 * (m2 + 2 * m3) * l2 * th2_dot**2 * np.sin(th2) - 0.5 * m3 * l3 * th3_dot**2 * np.sin(th3)
    )
    alphas[1, 0] = (0.5 * m2 + m3) * l2 * g * np.sin(th2) - 0.5 * m3 * l2 * l3 * th3_dot**2 * np.sin(th2 - th3)
    alphas[2, 0] = 0.5 * m3 * l3 * (g * np.sin(th3) + l2 * th2_dot**2 * np.sin(th2 - th3))

    mat = np.zeros((3, 3))
    mat[0, 0] = m1 + m2 + m3
    mat[1, 0] = -(0.5 * m2 + m3) * l2 * np.cos(th2)
    mat[2, 0] = -0.5 * m3 * l3 * np.cos(th3)
    mat[0, 1] = -(0.5 * m2 + m3) * l2 * np.cos(th2)
    mat[1, 1] = m3 * l2**2 + i2 + 0.25 * m2 * l2**2
    mat[2, 1] = 0.5 * m3 * l2 * l3 * np.cos(th2 - th3)
    mat[0, 2] = -0.5 * m3 * l3 * np.cos(th3)
    mat[1, 2] = 0.5 * m3 * l2 * l3 * np.cos(th2 - th3)
    mat[2, 2] = 0.25 * m2 * l3**2 + i3

    ddot = np.linalg.solve(mat, alphas)

    x_dot = np.vstack((x[3:, :], ddot))

    return x_dot


pend_derivX = lambda x, u: jit(jacfwd(jpendulum, 0))(x, u).reshape((6, 6))
pend_derivU = lambda x, u: jit(jacfwd(jpendulum, 1))(x, u).reshape((6, 1))


def cart_penulum_rk4(xk, uk, dt=DT):
    f1 = pendulum(xk, uk)
    f2 = pendulum(xk + f1 * dt / 2, uk)
    f3 = pendulum(xk + f2 * dt / 2, uk)
    f4 = pendulum(xk + f3 * dt, uk)

    return xk + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)


def visualize(lstates, lcontrols=None, legends=None, name=None):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    pad = 0.2
    if legends == None:
        legends = [""] * len(lstates)

    for k, states in enumerate(lstates):
        time = range(states.shape[1])
        time = [i * DT for i in time]

        cart_pos = states[0, :]
        # angles = np.arctan2(np.sin(states[1:3, :]), np.cos(states[1:3, :]))
        angles = states[1:3, :]

        ax1.plot(time, cart_pos, label=legends[k] + "x1")
        ax1.set_ylabel("x1")
        ax1.legend()

        ax2.plot(time, angles[0, :], label=legends[k] + "θ2")
        ax2.set_ylabel("θ2")
        ax2.legend()

        ax3.plot(time, angles[1, :], label=legends[k] + "θ3")
        ax3.set_ylabel("θ3")
        ax3.legend()

    plt.xlabel("Time (s)")
    plt.show(block=False)

    fig1 = None
    if lcontrols is not None:
        fig1, ax1 = plt.subplots()
        for k, controls in enumerate(lcontrols):
            max_u = np.max(np.abs(controls))
            ax1.plot(time, controls[0, :], label=legends[k] + "u")
            ax1.legend()
            ax1.set_ylim([-max_u - pad, max_u + pad])
            ax1.set_ylabel("u")
        plt.xlabel("Time (s)")
        plt.show(block=False)

    if name is not None:
        fig.savefig(name + "_qs")
        if fig1 is not None:
            fig1.savefig(name + "_us")


def animate(states, ref_state=None, init_state=False, show_trace=False):
    fig, ax = plt.subplots()

    size = 3

    ax.set_xlim(-size, size)
    ax.set_xticks(np.arange(-size, size, 0.5))
    ax.set_ylim(-size, size)
    ax.set_yticks(np.arange(-size, size, 0.5))

    if init_state:
        x0 = states[0, 0]
        x1 = x0 - l2 * np.sin(states[1, 0])
        y1 = +l2 * np.cos(states[1, 0])
        x2 = x1 - l3 * np.sin(states[2, 0])
        y2 = y1 + l3 * np.cos(states[2, 0])

        ax.plot([x0, x1, x2], [0, y1, y2], "o-", lw=1, color="black", label="initial state")
        ax.plot([x0 - 0.1, x0 + 0.1], [0, 0], "k-", lw=5)

    if ref_state is not None:
        x1 = x0 - l2 * np.sin(ref_state[1, 0])
        y1 = l2 * np.cos(ref_state[1, 0])
        x2 = x1 - l3 * np.sin(ref_state[2, 0])
        y2 = y1 + l3 * np.cos(ref_state[2, 0])

        ax.plot([x0, x1, x2], [0, y1, y2], "o-", lw=1, color="grey", label="reference state")
        ax.plot([x0 - 0.1, x0 + 0.1], [0, 0], "k-", lw=5)

    (cart,) = ax.plot([], [], "-", lw=5)
    (pend,) = ax.plot([], [], "o-", lw=1, label="current state")
    if show_trace:
        (trace,) = ax.plot([], [], lw=0.4, label="trace")
    x2_ = []
    y2_ = []

    ax.legend(fontsize=5)

    def update(frame):
        ret = []
        x0 = states[0, frame]
        x1 = x0 - l2 * np.sin(states[1, frame])
        y1 = l2 * np.cos(states[1, frame])
        x2 = x1 - l3 * np.sin(states[2, frame])
        y2 = y1 + l3 * np.cos(states[2, frame])

        x2_.append(x2)
        y2_.append(y2)

        pend.set_data([x0, x1, x2], [0, y1, y2])
        pend.set_zorder(2)
        cart.set_data([x0 - 0.1, x0 + 0.1], [0, 0])
        cart.set_zorder(1)
        if show_trace:
            trace.set_data(x2_, y2_)
            cart.set_zorder(0)
            return pend, cart, trace

        return pend, cart

    time_steps = states.shape[1]

    plt.gca().set_aspect("equal", adjustable="box")

    ani = FuncAnimation(fig, update, frames=time_steps, blit=True, interval=DT * 1000)

    plt.show()


if __name__ == "__main__":

    initial_state = np.array([[0, np.pi, np.pi - 0.3, 0, 0, 0]]).T
    u = np.array([0.0])

    # simulation
    T = 5
    t = 0

    states = initial_state
    x = np.copy(initial_state)

    t1 = time.time()

    while t < T:

        x = cart_penulum_rk4(x, u)

        states = np.hstack((states, x))

        t += DT

    print(time.time() - t1)
    # print(states.T)

    visualize([states])
    plt.show()
    # animate(states, show_trace=True, init_state=True)
