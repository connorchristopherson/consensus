import copy
import numpy as np
import matplotlib.pyplot as plt


def normalize_direction(u, v):
    normalized_u = u / np.sqrt(u**2 + v**2) + .00000001
    normalized_v = v / np.sqrt(u**2 + v**2) + .00000001
    return normalized_u, normalized_v


def plot_swarm(x, y, u, v, field_size, step, repulsion_radius,
               orientation_radius, attraction_radius):
    normalized_u, normalized_v = normalize_direction(u, v)
    plt.clf()
    plt.quiver(x, y, normalized_u, normalized_v)
    plt.scatter(x, y, color='b', s=20, marker="o")
    plt.axis([0, field_size, 0, field_size])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Step: {}".format(step))
    for i in range(len(x)):
        circle_r = plt.Circle((x[i], y[i]),
                              radius=repulsion_radius,
                              color='r',
                              fill=False)
        circle_o = plt.Circle((x[i], y[i]),
                              radius=repulsion_radius + orientation_radius,
                              color='y',
                              fill=False)
        circle_a = plt.Circle(
            (x[i], y[i]),
            radius=repulsion_radius + orientation_radius + attraction_radius,
            color='g',
            fill=False)
        plt.gcf().gca().add_artist(circle_r)
        plt.gcf().gca().add_artist(circle_o)
        plt.gcf().gca().add_artist(circle_a)

    plt.show(block=False)
    plt.pause(.6)


def point_orientation(x, y, u, v, base_index, neighbor_index, step):
    perfect_u = copy.deepcopy(u)
    perfect_v = copy.deepcopy(v)

    perfect_u[base_index] = x[neighbor_index] - x[base_index]
    perfect_v[base_index] = y[neighbor_index] - y[base_index]

    u[base_index] = (
        (1 - step) * u[base_index]) + (step * perfect_u[base_index])
    v[base_index] = (
        (1 - step) * v[base_index]) + (step * perfect_v[base_index])


def distance_formula(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def generate_As(x, y, repulsion_radius, orientation_radius, attraction_radius):
    num_agents = len(x)
    repulsion_A = np.zeros((num_agents, num_agents))
    orientation_A = np.zeros((num_agents, num_agents))
    attraction_A = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
            dist = distance_formula([x[i], y[i]], [x[j], y[j]])
            if dist <= repulsion_radius:
                repulsion_A[i, j] = 1.
            if dist <= repulsion_radius + orientation_radius and dist > repulsion_radius:
                orientation_A[i, j] = 1.
            if dist <= repulsion_radius + orientation_radius + attraction_radius and dist > repulsion_radius + orientation_radius:
                attraction_A[i, j] = 1.
    return repulsion_A, orientation_A, attraction_A


def generate_blind_spots(x, y, u, v, blind_spot, repulsion_A, orientation_A,
                         attraction_A):
    num_agents = len(x)
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
            pass  # generate blind spots


def create_diagonal(num_agents, A):
    diagonal = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        diagonal[i, i] = np.sum(A[i])
    return diagonal