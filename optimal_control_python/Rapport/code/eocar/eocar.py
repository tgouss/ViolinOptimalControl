import biorbd
from matplotlib import pyplot as plt

import biorbd_optim
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.dynamics import Dynamics

# --- Options --- #
# Model path
biorbd_model = biorbd.Model("eocar.bioMod")

# Problem parameters
number_shooting_points = 30
final_time = 2
ode_solver = biorbd_optim.OdeSolver.RK
is_cyclic_constraint, is_cyclic_objective = False, False

# Objective functions
objective_functions = ((ObjectiveFunction.minimize_torque, 100),)

# Dynamics
variable_type = biorbd_optim.Variable.variable_torque_driven
dynamics_func = Dynamics.forward_dynamics_torque_driven

# Constraints
constraints = (
    (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.START, (0, 1)),
    (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.END, (0, 2)),
)

# Define path constraint
X_bounds = biorbd_optim.Bounds()
X_init = biorbd_optim.InitialConditions()
ranges = []
for i in range(biorbd_model.nbSegment()):
    segRanges = biorbd_model.segment(i).ranges()
    for j in range(len(segRanges)):
        ranges.append(biorbd_model.segment(i).ranges()[j])

for i in range(biorbd_model.nbQ()):
    X_bounds.first_node_min.append(ranges[i].min())
    X_bounds.first_node_max.append(ranges[i].max())
    X_bounds.min.append(ranges[i].min())
    X_bounds.max.append(ranges[i].max())
    X_bounds.last_node_min.append(ranges[i].min())
    X_bounds.last_node_max.append(ranges[i].max())
    X_init.init.append(0)

for i in range(biorbd_model.nbQdot()):
    X_bounds.first_node_min.append(0)
    X_bounds.first_node_max.append(0)
    X_bounds.min.append(-15)
    X_bounds.max.append(15)
    X_bounds.last_node_min.append(0)
    X_bounds.last_node_max.append(0)
    X_init.init.append(0)

U_bounds = biorbd_optim.Bounds()
U_init = biorbd_optim.InitialConditions()
for i in range(biorbd_model.nbGeneralizedTorque()):
    U_bounds.min.append(-100)
    U_bounds.max.append(100)
    U_init.init.append(0)

# --- Solve the program --- #
nlp = biorbd_optim.OptimalControlProgram(
    biorbd_model,
    variable_type,
    dynamics_func,
    ode_solver,
    number_shooting_points,
    final_time,
    objective_functions,
    X_init,
    U_init,
    X_bounds,
    U_bounds,
    constraints,
    is_cyclic_constraint=is_cyclic_constraint,
    is_cyclic_objective=is_cyclic_objective,
)

sol = nlp.solve()
for idx in range(biorbd_model.nbQ()):
    plt.figure()
    q = sol["x"][0 * biorbd_model.nbQ() + idx :: 3 * biorbd_model.nbQ()]
    q_dot = sol["x"][1 * biorbd_model.nbQ() + idx :: 3 * biorbd_model.nbQ()]
    u = sol["x"][2 * biorbd_model.nbQ() + idx :: 3 * biorbd_model.nbQ()]
    plt.plot(q)
    plt.plot(q_dot)
    plt.plot(u)
plt.show()
