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

    diagonal_mask = np.eye(len(x)) == 1

    repulsion_A = np.where(np.less_equal(dist, rep_radius), 1., 0.)
    repulsion_A[diagonal_mask] = 0.

    orientation_A = np.where(
        np.logical_and(
            np.less_equal(dist, rep_radius + ori_radius),
            np.greater(dist, rep_radius)), 1., 0.)
    orientation_A[diagonal_mask] = 0.

    attraction_A = np.where(
        np.logical_and(
            np.less_equal(dist, rep_radius + ori_radius + att_radius),
            np.greater(dist, rep_radius + ori_radius)), 1., 0.)
    attraction_A[diagonal_mask] = 0.

    orientation_A[np.any(repulsion_A, axis=1)] = 0
    attraction_A[np.any(repulsion_A, axis=1)] = 0

    return repulsion_A, orientation_A, attraction_A


def create_diagonal(num_agents, A):
    diagonal = np.eye(num_agents)
    diagonal *= np.sum(A, axis=1)
    return diagonal


def generate_blind_spots(x, y, u, v, alpha, repulsion_A, orientation_A,
                         attraction_A):

    num_agents = x.shape[0]

    x2 = x.copy().reshape(-1, 1)
    x_diff = np.subtract(x2, x2.T)

    y2 = y.copy().reshape(-1, 1)
    y_diff = np.subtract(y2, y2.T)

    xy = np.stack([x_diff, y_diff])

    xy_norm = np.linalg.norm(xy, axis=0)
    xy = xy / xy_norm
    xy = np.nan_to_num(xy)
    xy = xy.reshape([2, -1])

    uv = np.stack([u, v])
    uv_norm = np.linalg.norm(uv, axis=0)
    uv = uv / uv_norm
    uv = np.nan_to_num(uv)
    uv = np.repeat(uv, num_agents, axis=1)

    angles = np.arccos(np.sum(xy * uv,
                              axis=0)).reshape([num_agents, num_agents])


    blind_filter = np.where(np.less(np.pi - angles, alpha / 2.), 1., 0.)

    repulsion_A = repulsion_A * blind_filter
    orientation_A = orientation_A * blind_filter
    attraction_A = attraction_A * blind_filter
    #print(blind_filter)
    return repulsion_A, orientation_A, attraction_A

def reorient_agents(x, y, u, v, repulsion_A, orientation_A,
    attraction_A, ORIENTATION_STEP_SIZE):
    uv = np.vstack((u, v)).T
    xy = np.vstack((x, y)).T

    xy_target = np.matmul(attraction_A, xy) - xy * \
                np.sum(attraction_A, axis=1).reshape(-1, 1)
    xy_target = xy_target / np.linalg.norm(xy_target, axis=1).reshape(-1, 1)
    xy_target = np.nan_to_num(xy_target)

    xy_repel = -np.matmul(repulsion_A, xy) + xy * \
                np.sum(repulsion_A, axis=1).reshape(-1, 1)
    xy_repel = xy_repel / np.linalg.norm(xy_repel, axis=1).reshape(-1, 1)
    xy_repel = np.nan_to_num(xy_repel)

    uv_target = np.matmul(orientation_A, uv)
    uv_target = uv_target / np.linalg.norm(uv_target, axis=1).reshape(-1, 1)
    uv_target = np.nan_to_num(uv_target)

    mixed_target = xy_target + uv_target
    mixed_target = mixed_target / np.linalg.norm(mixed_target, axis=1).reshape(-1, 1)

    has_repulsion = np.repeat(np.any(repulsion_A == 1, axis=1), repeats=2).reshape(-1, 2)
    has_orientation = np.repeat(np.any(orientation_A == 1, axis=1), repeats=2).reshape(-1, 2)
    has_attraction = np.repeat(np.any(attraction_A == 1, axis=1), repeats=2).reshape(-1, 2)

    final_target = np.where(has_repulsion,
                        xy_repel,
                        np.where(np.logical_and(has_orientation, has_attraction),
                            mixed_target,
                            np.where(has_attraction,
                                xy_target,
                                uv_target)))



    uv = (1 - ORIENTATION_STEP_SIZE) * uv + ORIENTATION_STEP_SIZE * final_target
    uv = uv / np.linalg.norm(uv, axis=1).reshape(-1, 1)
    u = uv[:, 0]
    v = uv[:, 1]
    return u, v