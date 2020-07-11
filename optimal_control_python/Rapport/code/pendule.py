from casadi import MX, Opti, vertcat, Function
from matplotlib import pyplot as plt
import numpy as np
import biorbd
import BiorbdViz

model = biorbd.Model("pendule.bioMod")


# Initialisation
T = 5  # 100 secondes
N = 30  # 30 intervalles
G = 9.81

continous_integration = True
number_of_nodes = 5
nbQ = model.nbQ()
nbQdot = model.nbQdot()
nbTau = model.nbGeneralizedTorque()
nx = nbQ + nbQdot
nu = nbTau
dt = T / N  # duree d'un intervalle
colors = ["k", "r", "g", "b", "y", "m"]
captions = [s.to_string() for s in model.nameDof()]

opti = Opti()
x = opti.variable(nx, N + 1)  # position et vitesse
u = opti.variable(nu, N)  # controle


# Declaration dynamique
x_sym = MX.sym("x", nx)
u_sym = MX.sym("u", nu)
dyn = Function(
    "Dynamics",
    [x_sym, u_sym],
    [
        vertcat(
            x_sym[nbQ:, 0],
            model.ForwardDynamics(
                x_sym[:nbQ, 0], x_sym[nbQ:, 0], u_sym
            ).to_mx(),
        )
    ],
    ["x", "u"],
    ["xdot"],
).expand()


def rk_calcul(x, u, dt_kutta):
    k1 = dyn(x, u)
    k2 = dyn(x + (dt_kutta / 2) * k1, u)
    k3 = dyn(x + (dt_kutta / 2) * k2, u)
    k4 = dyn(x + dt_kutta * k3, u)
    return x + (dt_kutta / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# RK4
dt_kutta = dt / (number_of_nodes)
rk = Function("RK", [x_sym, u_sym], [rk_calcul(x_sym, u_sym, dt_kutta)])
for j in range(N):
    x_next = x[:, j]
    for p in range(number_of_nodes):
        x_next = rk(x_next, u[:, j])

    # Contrainte de continuite
    opti.subject_to(x_next == x[:, j + 1])

# Fonction objectif
obj = MX(0)
for i in range(nu):
    for j in range(N):
        obj += u[i, j] * dt * u[i, j] * dt

opti.minimize(obj)  # optimisation sur plusieurs critères possible

# Contraintes de continuite
opti.subject_to(x[:, 0] == 0)  # vitesse et position nulles au debut
opti.subject_to(x[0, N] == 0)
opti.subject_to(x[1, N] == 3.14)
opti.subject_to(x[nbQ:, N] == 0)  # vitesses nulles à la fin

# Contraintes sur le chemin
for j in range(N + 1):
    opti.subject_to(x[:nbQ, j] >= -10)
    opti.subject_to(x[:nbQ, j] <= 20)
    opti.subject_to(x[nbQ:, j] >= -100)
    opti.subject_to(x[nbQ:, j] <= 100)

for j in range(N):
    opti.subject_to(u[0, j] >= -100)
    opti.subject_to(u[0, j] <= 100)
    opti.subject_to(u[1, j] == 0)

# Solution
opti.solver("ipopt")
sol = opti.solve() # appel au solveur

x_opt = sol.value(x)
u_opt = sol.value(u)

t_int = np.ndarray(((number_of_nodes - 1) * N + 2,))
q_int = np.ndarray((nbQ, (number_of_nodes - 1) * N + 2))
qdot_int = np.ndarray((nbQdot, (number_of_nodes - 1) * N + 2))
x_next = x_opt[:, 0]
cmp = 0
continous_integration = True
t_int[cmp] = 0
q_int[:, cmp] = x_next[:nbQ]
qdot_int[:, cmp] = x_next[nbQ:]
cmp += 1
for j in range(N):
    if not continous_integration:
        x_next = x_opt[:, j]

    for p in range(number_of_nodes):
        x_next = rk(x_next, u_opt[:, j])
        t_int[cmp] = dt_kutta * j * number_of_nodes + dt_kutta * p + dt_kutta
        q_int[:, cmp] = np.array(x_next[:nbQ, :]).reshape((nbQ,)).squeeze()
        qdot_int[:, cmp] = (
            np.array(x_next[nbQ:, :]).reshape((nbQdot,)).squeeze()
        )
        cmp += 1
    cmp -= 1

plt.subplot(221)
for i in range(nbQ):
    plt.title("Q (position) [états]")
    plt.plot(t_int, q_int[i, :], colors[i], label=captions[i])
    plt.xlabel("Time t")
    plt.ylabel("Position x")

plt.subplot(222)
for i in range(nbQdot):
    plt.title("Qdot (vitesse) [états]")
    plt.plot(t_int, qdot_int[i, :], colors[i], label=captions[i])
    plt.xlabel("Time t")
    plt.ylabel("Speed v")

plt.subplot(223)
for i in range(nu):
    plt.title("Tau (accélération) [commandes]")
    line = plt.step(
        np.linspace(0, T, N),
        u_opt[i],
        where="post",
        color=colors[i],
        label=captions[i],
    )
    plt.xlabel("Time t")
    plt.ylabel("Resultant of forces (control) u")


plt.tight_layout()
plt.legend()
plt.show()


BiorbdViz.BiorbdViz("pendule.bioMod").load_movement(q_int[:, :61])

