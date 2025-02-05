import numpy as np
import modeling as md
import traj_opt


sol = np.load("traj_opt_npy/run_0010.npy")

traj_states = traj_opt.extract_traj(sol)
traj_controls = traj_opt.extract_control(sol)
# print(traj_states)

md.visualize([traj_states], [traj_controls], name="opt_traj")
md.animate(traj_states, init_state=True, show_trace=True)
md.plt.show()
