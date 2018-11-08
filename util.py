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


def generate_As(x, y, rep_radius, ori_radius, att_radius):    
    dist_x = np.subtract(x.reshape(-1, 1), x.reshape(1, -1))
    dist_y = np.subtract(y.reshape(-1, 1), y.reshape(1, -1))
    dist = np.linalg.norm(np.stack([dist_x, dist_y]), axis=0)

    repulsion_A = np.where(np.less_equal(dist, rep_radius), 1., 0.)

    orientation_A = np.where(
                        np.logical_and(
                            np.less_equal(dist, rep_radius + ori_radius),
                            np.greater(dist, rep_radius) ),
                        1.,
                        0.)

    attraction_A = np.where(
                        np.logical_and(
                            np.less_equal(dist, rep_radius + ori_radius + att_radius),
                            np.greater(dist, rep_radius + ori_radius) ),
                        1.,
                        0.)

    anti_diag = 1. - np.diag(np.ones(repulsion_A.shape[0]))

    repulsion_A *= anti_diag
    orientation_A *= anti_diag
    attraction_A *= anti_diag

    return repulsion_A, orientation_A, attraction_A


def create_diagonal(num_agents, A):
    diagonal = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        diagonal[i, i] = np.sum(A[i])
    return diagonal


def generate_blind_spots(x, y, u, v, alpha, repulsion_A, orientation_A,
                         attraction_A):

    num_agents = len(x)

    x2 = x.copy().reshape(-1, 1)
    x_diff = np.subtract(x2, x2.T)

    y2 = y.copy().reshape(-1, 1)
    y_diff = np.subtract(y2, y2.T)

    xy = np.stack([x_diff, y_diff])
    xy_norm = np.linalg.norm(xy, axis=0)
    xy = xy / (xy_norm + .00000001)
    xy = xy.reshape([2, -1])

    uv = np.stack([u, v])
    uv_norm = np.linalg.norm(uv, axis=0)#.reshape()
    uv = uv / (uv_norm + .00000001)
    uv = np.repeat(uv, num_agents, axis=1)

    angles = np.arccos(np.sum(xy * uv, axis=0)).reshape([num_agents, num_agents])

    blind_filter = np.where(np.less(angles, alpha / 2.), 0., 1.)

    repulsion_A = repulsion_A * blind_filter
    orientation_A = orientation_A * blind_filter
    attraction_A = attraction_A * blind_filter
   
    return repulsion_A, orientation_A, attraction_A


def create_theta(u, v, x1, x2, y1, y2):
    x_sub = x2 - x1
    y_sub = y2 - y1
    neighbor_norm = np.sqrt(x_sub**2 + y_sub**2)
    uv_norm = np.sqrt(u**2 + v**2)
    dot_product = np.dot(np.array([x_sub, y_sub]), np.array([u, v]))
    return np.arccos(dot_product / (neighbor_norm * uv_norm))