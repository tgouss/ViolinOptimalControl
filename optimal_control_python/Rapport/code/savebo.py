# --- Save the optimal control program and the solution --- #
ocp.save(sol, "pendulum.bo", sol_iterations)

# --- Load the optimal control program and the solution --- #
ocp_load, sol_load = OptimalControlProgram.load("pendulum.bo")

# --- Show results --- #
result = ShowResult(ocp_load, sol_load)
result.graphs()
result.animate()
