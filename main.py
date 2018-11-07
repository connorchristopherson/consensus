import numpy as np
from util import *
from plotting import *

# hyperparamters
REPLUSION_RADIUS = 5
ORIENTATION_RADIUS = 3
ATTRACTION_RADIUS = 40
ORIENTATION_STEP_SIZE = .008
MOVEMENT_STEP_SIZE = .03
STEPS = 100
NUM_AGENTS = 10
FIELD_SIZE = 70
ALPHA = np.pi / 4.

# inits
x = np.random.randint(1, FIELD_SIZE - 1, size=NUM_AGENTS).astype(float)
y = np.random.randint(1, FIELD_SIZE - 1, size=NUM_AGENTS).astype(float)
u = np.random.choice(a=[0, 1], size=NUM_AGENTS, replace=True).astype(float)
v = np.random.choice(a=[0, 1], size=NUM_AGENTS, replace=True).astype(float)
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
    plot_radius=False)

# loop
for step in range(STEPS):
    # generate adjacency matrices
    repulsion_A, orientation_A, attraction_A = generate_As(
        x, y, REPLUSION_RADIUS, ORIENTATION_RADIUS, ATTRACTION_RADIUS)

    # generate blind spots
    repulsion_A, orientation_A, attraction_A = generate_blind_spots(
        x, y, u, v, ALPHA, repulsion_A, orientation_A, attraction_A)

    # reorient
    for base_index in range(len(x)):
        for neighbor_index in range(len(x)):
            if orientation_A[base_index, neighbor_index] == 1:
                point_orientation(
                    x,
                    y,
                    u,
                    v,
                    base_index=base_index,
                    neighbor_index=neighbor_index,
                    step=ORIENTATION_STEP_SIZE)

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
    x_dot = np.matmul(Lr, placements) - np.matmul(La, placements) + noise
    placements += x_dot * MOVEMENT_STEP_SIZE
    x = placements[:, 0]
    y = placements[:, 1]

    # plot
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
