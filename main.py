import numpy as np
from util import *
from plotting import *
import cProfile

# hyperparamters
NUM_AGENTS = 100
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 0
ATTRACTION_RADIUS = 50
ORIENTATION_STEP_SIZE = .05
MOVEMENT_STEP_SIZE = .05
STEPS = 500000
FIELD_SIZE = 30
ALPHA = (8 / 4) * np.pi
N_NEAREST = 5

# inits
x = np.random.randint(1, FIELD_SIZE - 1, size=NUM_AGENTS).astype(float)
y = np.random.randint(1, FIELD_SIZE - 1, size=NUM_AGENTS).astype(float)
u = np.random.random(NUM_AGENTS) * 2. - 1.
v = np.random.random(NUM_AGENTS) * 2. - 1.
uv = np.vstack((u, v)).T
uv = uv / np.linalg.norm(uv, axis=1).reshape(-1, 1)
u = uv[:, 0]
v = uv[:, 1]

replusion_eigen = []
orientation_eigen = []
attraction_eigen = []

# plot
plot(
    x,
    y,
    u,
    v,
    FIELD_SIZE,
    -1,
    REPLUSION_RADIUS,
    ORIENTATION_RADIUS,
    ATTRACTION_RADIUS,
    replusion_eigen,
    orientation_eigen,
    attraction_eigen,
    plot_radius=True)

# loop
for step in range(STEPS):
    # generate adjacency matrices
    repulsion_A, orientation_A, attraction_A = generate_As(
        x, y, REPLUSION_RADIUS, ORIENTATION_RADIUS, ATTRACTION_RADIUS, N_NEAREST)

    # generate blind spots
    repulsion_A, orientation_A, attraction_A = generate_blind_spots(
       x, y, u, v, ALPHA, repulsion_A, orientation_A, attraction_A)

    # reorient agents
    u, v = reorient_agents(x, y, u, v, repulsion_A,
        orientation_A, attraction_A, ORIENTATION_STEP_SIZE)

    # generate diagonals
    repulsion_diagonal = create_diagonal(NUM_AGENTS, repulsion_A)
    orientation_diagonal = create_diagonal(NUM_AGENTS, orientation_A)
    attraction_diagonal = create_diagonal(NUM_AGENTS, attraction_A)

    # generate Laplacians
    Lr = repulsion_diagonal - repulsion_A
    Lo = orientation_diagonal - orientation_A
    La = attraction_diagonal - attraction_A
    noise = (np.random.random(size=(NUM_AGENTS, 2)) - .5) * .01

    # generate Fiedler eigenvalues
    replusion_eigen.append(
        np.round(np.sort(np.linalg.eig(Lr)[0]), decimals=6)[1])
    orientation_eigen.append(
        np.round(np.sort(np.linalg.eig(Lo)[0]), decimals=6)[1])
    attraction_eigen.append(
        np.round(np.sort(np.linalg.eig(La)[0]), decimals=6)[1])

    # move
    placements = np.array(list(zip(x, y)))
    x_dot = np.array(list(zip(u, v))) + noise
    placements += x_dot * MOVEMENT_STEP_SIZE

    x = placements[:, 0]
    y = placements[:, 1]

    # plot
    if step % 11 == 0:
        plot(
            x,
            y,
            u,
            v,
            FIELD_SIZE,
            step,
            REPLUSION_RADIUS,
            ORIENTATION_RADIUS,
            ATTRACTION_RADIUS,
            replusion_eigen,
            orientation_eigen,
            attraction_eigen,
            plot_radius=False)


"""
TORUS:
# hyperparamters
NUM_AGENTS = 100
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 3
ATTRACTION_RADIUS = 50
ORIENTATION_STEP_SIZE = .01
MOVEMENT_STEP_SIZE = .01
STEPS = 500000
FIELD_SIZE = 50
ALPHA = (6 / 4) * np.pi

LOOSELY CORRELATED:
# hyperparamters
NUM_AGENTS = 100
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 7
ATTRACTION_RADIUS = 50
ORIENTATION_STEP_SIZE = .01
MOVEMENT_STEP_SIZE = .01
STEPS = 500000
FIELD_SIZE = 50
ALPHA = (8 / 4) * np.pi

PARALLEL:
# hyperparamters
NUM_AGENTS = 100
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 20
ATTRACTION_RADIUS = 50
ORIENTATION_STEP_SIZE = .01
MOVEMENT_STEP_SIZE = .01
STEPS = 500000
FIELD_SIZE = 50
ALPHA = (5 / 4) * np.pi

SWARM:
# hyperparamters
NUM_AGENTS = 100
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 0
ATTRACTION_RADIUS = 50
ORIENTATION_STEP_SIZE = .01
MOVEMENT_STEP_SIZE = .01
STEPS = 500000
FIELD_SIZE = 50
ALPHA = (8 / 4) * np.pi

NN_HIGHLY PARALLEL
# hyperparamters
NUM_AGENTS = 100
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 40
ATTRACTION_RADIUS = 50
ORIENTATION_STEP_SIZE = .05
MOVEMENT_STEP_SIZE = .05
STEPS = 500000
FIELD_SIZE = 30
ALPHA = (6 / 4) * np.pi
N_NEAREST = 5
"""