import numpy as np
from util import *
from plotting import *
import cProfile

# hyperparamters
NUM_AGENTS = 50
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 0
ATTRACTION_RADIUS = 10
ORIENTATION_STEP_SIZE = .01  #/ NUM_AGENTS
MOVEMENT_STEP_SIZE = .01  #/ NUM_AGENTS
STEPS = 500000
FIELD_SIZE = 50
ALPHA = (5 / 4) * np.pi

# inits
x = np.random.randint(1, FIELD_SIZE - 1 - 10, size=NUM_AGENTS).astype(float)
y = np.random.randint(1, FIELD_SIZE - 1 - 10, size=NUM_AGENTS).astype(float)
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
        x, y, REPLUSION_RADIUS, ORIENTATION_RADIUS, ATTRACTION_RADIUS)

    # generate blind spots
    repulsion_A, orientation_A, attraction_A = generate_blind_spots(
       x, y, u, v, ALPHA, repulsion_A, orientation_A, attraction_A)

    # reorient agents
    u, v = reorient_agents(x, y, u, v, repulsion_A,
        orientation_A, attraction_A, ORIENTATION_STEP_SIZE)

    # MODIFY RED AND GOLD LEADER ORIENTATION

    # generate diagonals
    repulsion_diagonal = create_diagonal(NUM_AGENTS, repulsion_A)
    orientation_diagonal = create_diagonal(NUM_AGENTS, orientation_A)
    attraction_diagonal = create_diagonal(NUM_AGENTS, attraction_A)

    # generate Laplacians
    #print(attraction_A)
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
    x_dot = np.array(list(zip(u, v)))
    placements += x_dot * MOVEMENT_STEP_SIZE

    x = placements[:, 0]
    y = placements[:, 1]

    # ZERO OUT THE FIRST TWO ROWS OF XDOT AND IMPLEMENT RED AND GOLD LEADER MOVEMENT

    # plot
    if step % 21 == 0:
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
