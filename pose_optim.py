import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_interp_spline
from scipy.interpolate import make_lsq_spline, BSpline, LSQUnivariateSpline
from scipy.optimize import minimize
import numpy as np
import json
'''Process the raw triangulation from multiview calib.'''

def to_mat(data):
    n_frames = len(data.keys())
    mat = np.full((n_frames, 25, 3), np.inf)

    for frame in range(n_frames):
        xyz = data[f"frame{frame}"]["points_3d"]
        ids = data[f"frame{frame}"]["ids"]
        for i, id in enumerate(ids):
            mat[frame][id] = xyz[i]

    return mat

def compute_seg_lengths(joints, start_indices, end_indices):

    seg_nbr = len(start_indices)
    seg_lengths = np.zeros(seg_nbr)

    for i in range(seg_nbr):
        seg_lengths[i] = np.linalg.norm((joints[start_indices[i]] - joints[end_indices[i]]))

    return seg_lengths


def compute_avg_seg_lengths(mat, start_indices, end_indices):
    seg_nbr = len(start_indices)
    n_frames, n_joints, n_dim = mat.shape
    seg_len_mat = np.zeros((n_frames, seg_nbr))
    for i in range(n_frames):
        seg_len_mat[i] = compute_seg_lengths(mat[i], start_indices, end_indices)

    avg_seg_lengths = np.zeros(seg_nbr)
    for i in range(seg_nbr):
        seg_len = seg_len_mat[:, i]
        avg_seg_lengths[i] = np.average(seg_len[seg_len < 1500])

    return avg_seg_lengths # np.average(seg_len_mat, axis=0)


def seg_symmetry(seg_lengths):
    sym_mat = [[1, 4], [2, 5], [3, 6], [20, 22], [21, 23], [8, 14], [9, 15], [10, 16], [11, 17], [12, 18], [13, 19]]

    for sym_ind in sym_mat:
        seg_avg = (seg_lengths[sym_ind[0]] + seg_lengths[sym_ind[1]]) / 2
        seg_lengths[sym_ind[0]] = seg_avg
        seg_lengths[sym_ind[1]] = seg_avg

    return seg_lengths

def seg_distance(joints, avg_seg_lengths, start_indices, end_indices):
    seg_lengths = compute_seg_lengths(joints, start_indices, end_indices)
    r = np.linalg.norm(seg_lengths - avg_seg_lengths)

    return r

def hips_loss(joints, w = 1):
    '''hips joints loss term'''
    left_j = 12
    right_j = 9
    center_j = 8
    up_j = 1

    v1 = (joints[up_j] - joints[center_j]) / np.linalg.norm(joints[up_j] - joints[center_j])
    v2 = (joints[left_j] - joints[center_j]) / np.linalg.norm(joints[left_j] - joints[center_j])
    v3 = (joints[right_j] - joints[center_j]) / np.linalg.norm(joints[right_j] - joints[center_j])

    c1 = np.cross(v1, v2)
    c2 = np.cross(v3, v1)
    l = np.dot(c1, c2)

    return  np.exp(w * (l - 1)**2) - 1


def shoulder_loss(joints, w = 1):
    '''Shoulder joints loss term'''
    left_j = 5
    right_j = 2
    center_j = 1
    down_j = 8

    v1 = (joints[down_j] - joints[center_j]) / np.linalg.norm(joints[down_j] - joints[center_j])
    v2 = (joints[left_j] - joints[center_j]) / np.linalg.norm(joints[left_j] - joints[center_j])
    v3 = (joints[right_j] - joints[center_j]) / np.linalg.norm(joints[right_j] - joints[center_j])

    c1 = np.cross(v1, v2)
    c2 = np.cross(v3, v1)
    l = np.dot(c1, c2)

    return  np.exp(w * (l - 1)**2) - 1


def reg_seg_distance(joints, joints_init, avg_seg_lengths, start_indices, end_indices):
    dl = seg_distance(joints.reshape(25, 3), avg_seg_lengths, start_indices, end_indices)
    regl = 0.0001 * np.linalg.norm(joints - joints_init.reshape(75), 1)
    hl = 1 * hips_loss(joints.reshape(25, 3))
    sl = 1 * shoulder_loss(joints.reshape(25, 3))
    
    return dl + regl + hl + sl


def spline_interpolation(mat, knots_dist):
    n_frames, n_joints, n_dim = mat.shape
    x = np.arange(n_frames)
    t = list(x[::knots_dist])
    t = t[1:]  # interior points

    for joint in range(n_joints):
        for dim in range(n_dim):
            y = mat[:, joint, dim]
            w = np.isinf(y)
            y[w] = 0

            cs = LSQUnivariateSpline(x, y, t, w=~w)
            mat[:, joint, dim] = cs(x)

    return mat

def remove_outliers(mat, max_dist = 3000):
    n_frames, n_joints, n_dim = mat.shape
    for frame in range(n_frames):
        for joint in range(n_joints):
            d = np.linalg.norm(mat[frame][joint])
            if d > max_dist:
                mat[frame][joint] = np.full(3, np.inf)

    return mat


with open('all_frames_sub1.json') as file:
    data1 = json.load(file)

with open('all_frames_sub2.json') as file:
    data2 = json.load(file)

mat1 = to_mat(data1)
mat2 = to_mat(data2)
mat1 = remove_outliers(mat1)
mat2 = remove_outliers(mat2)
mat1 = spline_interpolation(mat1, 30)
mat2 = spline_interpolation(mat2, 30)

n_frames, n_joints, n_dim = mat1.shape
x = np.arange(n_frames)

start_indices = np.array([0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 11, 11, 22, 8, 12, 13, 14, 14, 19, 0, 15, 0, 16])
end_indices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 24, 22, 23, 12, 13, 14, 21, 19, 20, 15, 17, 16, 18])

avg_seg_len_1 = compute_avg_seg_lengths(mat1, start_indices, end_indices)
avg_seg_len_2 = compute_avg_seg_lengths(mat2, start_indices, end_indices)
avg_seg_len_1 = seg_symmetry(avg_seg_len_1)
avg_seg_len_2 = seg_symmetry(avg_seg_len_2)

for frame in range(n_frames):
    print(f'frame : {frame}')
    res = minimize(lambda j : reg_seg_distance(j, mat1[frame], avg_seg_len_1, start_indices, end_indices), mat1[frame],
                   method='SLSQP', tol = 0.001, options={'disp': True, 'maxiter': 300})
    mat1[frame] = res.x.reshape(25, 3)
    res = minimize(lambda j : reg_seg_distance(j, mat2[frame], avg_seg_len_2, start_indices, end_indices), mat2[frame],
                   method='SLSQP', tol = 0.001, options={'disp': True, 'maxiter': 300})
    mat2[frame] = res.x.reshape(25, 3)


mat1 = spline_interpolation(mat1, 30)
mat2 = spline_interpolation(mat2, 30)

dict = {}
dict['p1'] = mat1.tolist()
dict['p2'] = mat2.tolist()
with open('seq7_poses_5.json', 'w') as file:
    json.dump(dict, file)
