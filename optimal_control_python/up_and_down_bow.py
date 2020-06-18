import time

import biorbd
import numpy as np
from casadi import MX, vertcat

from biorbd_optim import (
    Instant,
    InterpolationType,
    Axe,
    OptimalControlProgram,
    Dynamics,
    ProblemType,
    Objective,
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
)

from utils import Bow, Violin, Muscles


def xia_model_dynamic(states, controls, parameters, nlp):
    Dynamics.apply_parameters(parameters, nlp)
    nbq = nlp["model"].nbQ()
    nbqdot = nlp["model"].nbQdot()
    nb_q_and_qdot = nbq + nbqdot

    q = states[:nbq]
    qdot = states[nbq:nb_q_and_qdot]
    activate = states[nb_q_and_qdot : nb_q_and_qdot + nlp["nbMuscle"]]
    fatigue = states[nb_q_and_qdot + nlp["nbMuscle"] : nb_q_and_qdot + 2 * nlp["nbMuscle"]]
    resting = states[nb_q_and_qdot + 2 * nlp["nbMuscle"] :]

    muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])

    residual_tau = controls[: nlp["nbTau"]]
    excitement = controls[nlp["nbTau"] :]

    for k in range(nlp["nbMuscle"]):
        resting[k] += fatigue[k] * Muscles.R
        fatigue[k] -= fatigue[k] * Muscles.R

        fatigue[k] += activate[k] * Muscles.F
        activate[k] -= activate[k] * Muscles.F

        resting[k] += activate[k]  # catch not use fiber

        muscles_states[k].setActivation(activate[k])
        muscles_states[k].setActivation(excitement[k])

        # todo fix force max


    activate_dot = nlp["model"].activationDot(muscles_states).to_mx()
    muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
    tau = muscles_tau + residual_tau

    qdot_reduced = nlp["q_mapping"].reduce.map(qdot)
    dxdt = MX(nlp["nx"], nlp["ns"])
    for i, f_ext in enumerate(nlp["external_forces"]):
        qddot = biorbd.Model.ForwardDynamics(
            nlp["model"], q, qdot, tau, f_ext
        ).to_mx()  # ForwardDynamicsConstraintsDirect
        dxdt[:, i] = vertcat(qdot, qddot, activate_dot)

    return dxdt


def prepare_nlp(biorbd_model_path="../models/BrasViolon.bioMod"):
    """
    Mix .bioMod and users data to call OptimalControlProgram constructor.
    :param biorbd_model_path: path to the .bioMod file.
    :param show_online_optim: bool which active live plot function.
    :return: OptimalControlProgram object.
    """
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5
    torque_min, torque_max, torque_init = -100, 100, 0

    # Problem parameters
    number_shooting_points = 30
    final_time = 0.5

    # Choose the string of the violin
    violon_string = Violin("G")
    inital_bow_side = Bow("frog")

    # Add objective functions
    objective_functions = (
        {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 100},
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1},
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "controls_idx": [0, 1, 2, 3], "weight": 2000},
    )

    # Dynamics
    problem_type = {"type": ProblemType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN, "dynamic": custom_dynamic}


    # Constraints
    constraints = (
        {
            "type": Constraint.ALIGN_MARKERS,
            "instant": Instant.START,
            "first_marker_idx": Bow.frog_marker,
            "second_marker_idx": violon_string.bridge_marker,
        },
        {
            "type": Constraint.ALIGN_MARKERS,
            "instant": Instant.MID,
            "first_marker_idx": Bow.tip_marker,
            "second_marker_idx": violon_string.bridge_marker,
        },
        {
            "type": Constraint.ALIGN_MARKERS,
            "instant": Instant.END,
            "first_marker_idx": Bow.frog_marker,
            "second_marker_idx": violon_string.bridge_marker,
        },
        {
            "type": Constraint.ALIGN_SEGMENT_WITH_CUSTOM_RT,
            "instant": Instant.ALL,
            "segment_idx": Bow.segment_idx,
            "rt_idx": violon_string.rt_on_string,
        },
        {
            "type": Constraint.ALIGN_MARKER_WITH_SEGMENT_AXIS,
            "instant": Instant.ALL,
            "marker_idx": violon_string.bridge_marker,
            "segment_idx": Bow.segment_idx,
            "axis": (Axe.Y),
        },
        {
            "type": Constraint.ALIGN_MARKERS,
            "instant": Instant.ALL,
            "first_marker_idx": Bow.contact_marker,
            "second_marker_idx": violon_string.bridge_marker,
        },
        # TODO: add constraint about velocity in a marker of bow (start and end instant)
    )

    # External forces
    external_forces = [np.repeat(violon_string.external_force[:, np.newaxis], number_shooting_points, axis=1)]

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    for k in range(biorbd_model.nbQ(), biorbd_model.nbQdot()):
        X_bounds.first_node_min[k] = 0
        X_bounds.first_node_max[k] = 0
        X_bounds.last_node_min[k] = 0
        X_bounds.last_node_max[k] = 0

    # Initial guess

    optimal_initial_values = True
    if optimal_initial_values:
        X_init = InitialConditions(violon_string.x_init, InterpolationType.EACH_FRAME)
        U_init = InitialConditions(violon_string.u_init, InterpolationType.EACH_FRAME)
    else:
        X_init = InitialConditions(
            violon_string.initial_position()[inital_bow_side.side] + [0] * biorbd_model.nbQdot(),
            InterpolationType.CONSTANT,
        )
        U_init = InitialConditions(
            [torque_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscleTotal(),
            InterpolationType.CONSTANT,
        )

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
    )

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        external_forces=external_forces,
        nb_threads=4,
    )


if __name__ == "__main__":
    ocp = prepare_nlp()

    # --- Solve the program --- #
    tic = time.time()
    sol, sol_iterations = ocp.solve(
        show_online_optim=True,
        return_iterations=True,
        options_ipopt={"tol": 1e-4, "max_iter": 2000, "ipopt.bound_push": 1e-10, "ipopt.bound_frac": 1e-10},
    )
    toc = time.time() - tic
    print(f"Time to solve : {toc}sec")

    t = time.localtime(time.time())
    date = f"{t.tm_year}_{t.tm_mon}_{t.tm_mday}"
    OptimalControlProgram.save(ocp, sol, f"results/{date}_upDown.bo")
    OptimalControlProgram.save_get_data(ocp, sol, f"results/{date}_upDown.bob", sol_iterations=sol_iterations)
    OptimalControlProgram.save_get_data(ocp, sol, f"results/{date}_upDown_interpolate.bob", interpolate_nb_frames=100)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
