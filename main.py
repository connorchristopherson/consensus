import numpy as np
from util import *
from plotting import *
import cProfile

# hyperparamters
REPLUSION_RADIUS = 5
ORIENTATION_RADIUS = 10
ATTRACTION_RADIUS = 10
ORIENTATION_STEP_SIZE = .008
MOVEMENT_STEP_SIZE = .001
STEPS = 5000
NUM_AGENTS = 50
FIELD_SIZE = 150
ALPHA = (6 / 4) * np.pi

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
        x, y, REPLUSION_RADIUS, ORIENTATION_RADIUS, ATTRACTION_RADIUS)

    # generate blind spots
    repulsion_A, orientation_A, attraction_A = generate_blind_spots(
        x, y, u, v, ALPHA, repulsion_A, orientation_A, attraction_A)

    # reorient agents
    uv = np.vstack((u, v)).T
    xy = np.vstack((x, y)).T

    xy_target = np.matmul(attraction_A, xy) - xy * \
                np.sum(attraction_A, axis=1).reshape(-1, 1)
    xy_target = xy_target / np.linalg.norm(xy_target, axis=1).reshape(-1, 1)
    xy_target = np.nan_to_num(xy_target)

    uv_target = np.matmul(orientation_A, uv)
    uv_target = uv_target / np.linalg.norm(uv_target, axis=1).reshape(-1, 1)
    uv_target = np.nan_to_num(uv_target)

    mixed_target = xy_target + uv_target
    mixed_target = np.linalg.norm(mixed_target, axis=1).reshape(-1, 1)

    final_target = np.where(np.equal(orientation_A, attraction_A))

    uv = np.where(
        np.all(attraction_A + orientation_A > 1.,
               axis=1).reshape(-1, 1), ((1 - ORIENTATION_STEP_SIZE) * uv +
                                        ORIENTATION_STEP_SIZE * mixed_target),
        ((1 - ORIENTATION_STEP_SIZE) * uv + ORIENTATION_STEP_SIZE * xy_target +
         ORIENTATION_STEP_SIZE * uv_target))

    uv = uv / np.linalg.norm(uv, axis=1).reshape(-1, 1)
    u = uv[:, 0]
    v = uv[:, 1]

    # MODIFY RED AND GOLD LEADER ORIENTATION

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

    # ZERO OUT THE FIRST TWO ROWS OF XDOT AND IMPLEMENT RED AND GOLD LEADER MOVEMENT

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
            plot_radius=True)
