@staticmethod
def forward_dynamics_torque_muscle_driven(states, controls, nlp):
    q, qdot, residual_tau = Dynamics.__dispatch_data(states, controls, nlp)

    muscles_states = biorbd.VecBiorbdMuscleStateDynamics(nlp["nbMuscle"])
    muscles_activations = controls[nlp["nbTau"] :]
    for k in range(nlp["nbMuscle"]):
        muscles_states[k].setActivation(muscles_activations[k])
    muscles_tau = (
        nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
    )

    tau = muscles_tau + residual_tau
    qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau).to_mx()
    return vertcat(qdot_reduced, qddot_reduced)
