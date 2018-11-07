import copy
import numpy as np


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


# def generate_blind_spots(x, y, u, v, blind_spot, repulsion_A, orientation_A,
#                          attraction_A):
#     num_agents = len(x)
#     for i in range(num_agents):
#         for j in range(num_agents):
#             if i == j:
#                 continue
#             pass  # generate blind spots


def create_diagonal(num_agents, A):
    diagonal = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        diagonal[i, i] = np.sum(A[i])
    return diagonal


def generate_blind_spots(x, y, u, v, alpha, repulsion_A, orientation_A,
                         attraction_A):
    num_agents = len(x)
    for i in range(num_agents):
        direction_theta = generate_theta(0., u[i], 0., v[i]) % (2. * np.pi)
        for j in range(num_agents):
            if i == j:
                continue
            neighbor_theta = (generate_theta(
                x[i], x[j], y[i], y[j]) - direction_theta) % (2. * np.pi)

            if neighbor_theta >= np.pi - (
                    alpha / 2.) and neighbor_theta <= np.pi - (alpha / 2.):
                repulsion_A[i, j] = 0.
                orientation_A[i, j] = 0.
                attraction_A[i, j] = 0.
    return repulsion_A, orientation_A, attraction_A


# x = [0, u]
# y = [0, v]
# def f2(u, v):
#     num_agents = len(u)
#     for i in range(num_agents):
#         for j in range(num_agents):
#             if i == j:
#                 continue
#             denom = np.sqrt(u[j]**2 + v[j]**2)
#             quotient = u[j] / denom
#             theta = np.arccos(quotient)


def generate_theta(x1, x2, y1, y2):
    x_sub = x2 - x1
    y_sub = y2 - y1
    denom = np.sqrt(x_sub**2 + y_sub**2)
    quotient = x_sub / denom
    return np.arccos(quotient)