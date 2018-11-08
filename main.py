import numpy as np
from util import *
from plotting import *
import cProfile

# hyperparamters
REPLUSION_RADIUS = 20
ORIENTATION_RADIUS = 5
ATTRACTION_RADIUS = 4
ORIENTATION_STEP_SIZE = .008
MOVEMENT_STEP_SIZE = .0003
STEPS = 1000
NUM_AGENTS = 100
FIELD_SIZE = 70
ALPHA = (4/4)*np.pi

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
    plot_radius=False)

#pr = cProfile.Profile()
#pr.enable()

# loop
for step in range(STEPS):
    # generate adjacency matrices
    repulsion_A, orientation_A, attraction_A = generate_As(
        x, y, REPLUSION_RADIUS, ORIENTATION_RADIUS, ATTRACTION_RADIUS)

    # generate blind spots
    repulsion_A, orientation_A, attraction_A = generate_blind_spots(
        x, y, u, v, ALPHA, repulsion_A, orientation_A, attraction_A)
    """
    # reorient in attration region
    for base_index in range(len(x)):
        for neighbor_index in range(len(x)):
            if attraction_A[base_index, neighbor_index] == 1:
                point_orientation(
                    x,
                    y,
                    u,
                    v,
                    base_index=base_index,
                    neighbor_index=neighbor_index,
                    step=ORIENTATION_STEP_SIZE)
    """
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

    uv = np.where(np.all(attraction_A + orientation_A > 1., axis=1).reshape(-1, 1),
        ((1 - ORIENTATION_STEP_SIZE ) * uv
            + ORIENTATION_STEP_SIZE * mixed_target),
        ((1 - ORIENTATION_STEP_SIZE ) * uv
            + ORIENTATION_STEP_SIZE * xy_target
            + ORIENTATION_STEP_SIZE * uv_target))

    uv = uv / np.linalg.norm(uv, axis=1).reshape(-1, 1)
    u = uv[:, 0]
    v = uv[:, 1]

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
    if step % 500 == 0:
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
        

#pr.disable()
#pr.print_stats()