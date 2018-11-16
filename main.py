import numpy as np
from util import *
from plotting import *
import cProfile
from tqdm import tqdm

# hyperparamters
NUM_AGENTS = 100
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 3
ATTRACTION_RADIUS = 50
ORIENTATION_STEP_SIZE = .01
MOVEMENT_STEP_SIZE = .01
FIELD_SIZE = 50
ALPHA = (6 / 4) * np.pi
N_NEAREST = -1

RUNS = 1
STEPS = 10000
SHOULD_PLOT_AGENTS = True

eigenvalues_runs = np.empty((RUNS, STEPS))

for i in range(RUNS):
    # inits
    x = np.random.randint(1, FIELD_SIZE - 1, size=NUM_AGENTS).astype(float)
    y = np.random.randint(1, FIELD_SIZE - 1, size=NUM_AGENTS).astype(float)
    u = np.random.random(NUM_AGENTS) * 2. - 1.
    v = np.random.random(NUM_AGENTS) * 2. - 1.
    uv = np.vstack((u, v)).T
    uv = uv / np.linalg.norm(uv, axis=1).reshape(-1, 1)
    u = uv[:, 0]
    v = uv[:, 1]

    # replusion_eigen = []
    orientation_eigen = []
    attraction_eigen = []
    lall_eigen = []

    if SHOULD_PLOT_AGENTS:
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
            lall_eigen,
            plot_radius=True)

    # loop
    for step in tqdm(range(STEPS)):
        # generate adjacency matrices
        repulsion_A, orientation_A, attraction_A = generate_As(
            x, y, REPLUSION_RADIUS, ORIENTATION_RADIUS, ATTRACTION_RADIUS, N_NEAREST)

        # generate blind spots
        repulsion_A, orientation_A, attraction_A = generate_blind_spots(
           x, y, u, v, ALPHA, repulsion_A, orientation_A, attraction_A)

        # reorient agents
        u, v = reorient_agents(x, y, u, v, repulsion_A,
            orientation_A, attraction_A, ORIENTATION_STEP_SIZE)

        # make repulsion bidirectional
        repulsion_A_flipped = np.flipud(np.rot90(repulsion_A))
        repulsion_A = np.logical_or(repulsion_A == 1, repulsion_A_flipped == 1).astype(int)

        # make orientation bidirectional
        orientation_A_flipped = np.flipud(np.rot90(orientation_A))
        orientation_A = np.logical_or(orientation_A == 1, orientation_A_flipped == 1).astype(int)

        # make attraction bidirectional
        attraction_A_flipped = np.flipud(np.rot90(attraction_A))
        attraction_A = np.logical_or(attraction_A == 1, attraction_A_flipped == 1).astype(int)

        # generate diagonals
        repulsion_diagonal = create_diagonal(NUM_AGENTS, repulsion_A)
        orientation_diagonal = create_diagonal(NUM_AGENTS, orientation_A)
        attraction_diagonal = create_diagonal(NUM_AGENTS, attraction_A)

        # generate Laplacians
        Lr = repulsion_diagonal - repulsion_A
        Lo = orientation_diagonal - orientation_A
        La = attraction_diagonal - attraction_A
        Lall = Lr + Lo + La
        # print(repulsion_A)
        # print(orientation_A)
        # print(attraction_A)
        # print(Lall)
        # print()
        # print()
        noise = (np.random.random(size=(NUM_AGENTS, 2)) - .5) * .01

        # generate Fiedler eigenvalues
        """
        replusion_eigen.append(
            np.round(np.sort(np.linalg.eig(Lr)[0]), decimals=6)[1])
        orientation_eigen.append(
            np.round(np.sort(np.linalg.eig(Lo)[0]), decimals=6)[1])
        attraction_eigen.append(
            np.round(np.sort(np.linalg.eig(La)[0]), decimals=6)[1])
        """
        
        # lall_eigen.append(
        #     np.round(np.sort(np.linalg.eig(Lall)[0]), decimals=6)[1])
        
        lall_eigen = np.ones(Lall.shape[0])
        # move
        placements = np.array(list(zip(x, y)))
        x_dot = np.array(list(zip(u, v))) + noise
        placements += x_dot * MOVEMENT_STEP_SIZE

        x = placements[:, 0]
        y = placements[:, 1]

        if SHOULD_PLOT_AGENTS and step % 21 == 0:
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
                lall_eigen,
                plot_radius=False)

    eigenvalues_runs[i] = np.asarray(lall_eigen)


plot_runs_eigenvalues(eigenvalues_runs)

"""
TORUS:
# hyperparamters
NUM_AGENTS = 100
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 3
ATTRACTION_RADIUS = 50
ORIENTATION_STEP_SIZE = .01
MOVEMENT_STEP_SIZE = .01
FIELD_SIZE = 50
ALPHA = (6 / 4) * np.pi
N_NEAREST = -1

LOOSELY CORRELATED:
# hyperparamters
NUM_AGENTS = 100
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 7
ATTRACTION_RADIUS = 50
ORIENTATION_STEP_SIZE = .01
MOVEMENT_STEP_SIZE = .01
FIELD_SIZE = 50
ALPHA = (8 / 4) * np.pi
N_NEAREST = -1

PARALLEL:
# hyperparamters
NUM_AGENTS = 100
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 20
ATTRACTION_RADIUS = 50
ORIENTATION_STEP_SIZE = .01
MOVEMENT_STEP_SIZE = .01
FIELD_SIZE = 50
ALPHA = (5 / 4) * np.pi
N_NEAREST = -1

SWARM:
# hyperparamters
NUM_AGENTS = 100
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 0
ATTRACTION_RADIUS = 50
ORIENTATION_STEP_SIZE = .01
MOVEMENT_STEP_SIZE = .01
FIELD_SIZE = 50
ALPHA = (8 / 4) * np.pi
N_NEAREST = -1

NN_HIGHLY PARALLEL
# hyperparamters
NUM_AGENTS = 100
REPLUSION_RADIUS = 1
ORIENTATION_RADIUS = 40
ATTRACTION_RADIUS = 50
ORIENTATION_STEP_SIZE = .05
MOVEMENT_STEP_SIZE = .05
FIELD_SIZE = 30
ALPHA = (6 / 4) * np.pi
N_NEAREST = 5
"""