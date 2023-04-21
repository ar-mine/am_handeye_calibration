#!/usr/bin/env python
# coding: utf-8
import transforms3d as tfs
import numpy as np
import math
import pickle


def get_matrix_eular_radu(x, y, z, rx, ry, rz):
    rmat = tfs.euler.euler2mat(math.radians(rx), math.radians(ry), math.radians(rz))
    rmat = tfs.affines.compose(np.squeeze(np.asarray((x, y, z))), rmat, [1, 1, 1])
    return rmat


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def rot2quat_minimal(m):
    quat = tfs.quaternions.mat2quat(m[0:3, 0:3])
    return quat[1:]


def quatMinimal2rot(q):
    p = np.dot(q.T, q)
    w = np.sqrt(np.subtract(1, p[0][0]))
    return tfs.quaternions.quat2mat([w, q[0], q[1], q[2]])


with open("./calibration.pkl", 'rb') as f:
    data = pickle.load(f)

board_R_list = data['board_R_list']
board_T_list = data['board_T_list']
eef_R_list = data['eef_R_list']
eef_T_list = data['eef_T_list']

Hgs, Hcs = [], []
for i in range(len(board_R_list)):
    board_transform = np.eye(4)
    board_transform[:3, :3] = board_R_list[i]
    board_transform[:3, 3] = board_T_list[i]

    eef_transform = np.eye(4)
    eef_transform[:3, :3] = eef_R_list[i]
    eef_transform[:3, 3] = eef_T_list[i]

    Hgs.append(eef_transform)
    Hcs.append(board_transform)

Hgijs = []
Hcijs = []
A = []
B = []
size = 0
for i in range(len(Hgs)):
    for j in range(i + 1, len(Hgs)):
        size += 1
        Hgij = np.dot(np.linalg.inv(Hgs[j]), Hgs[i])
        Hgijs.append(Hgij)
        Pgij = np.dot(2, rot2quat_minimal(Hgij))

        Hcij = np.dot(Hcs[j], np.linalg.inv(Hcs[i]))
        Hcijs.append(Hcij)
        Pcij = np.dot(2, rot2quat_minimal(Hcij))

        A.append(skew(np.add(Pgij, Pcij)))
        B.append(np.subtract(Pcij, Pgij))
MA = np.asarray(A).reshape(size * 3, 3)
MB = np.asarray(B).reshape(size * 3, 1)
Pcg_ = np.dot(np.linalg.pinv(MA), MB)
pcg_norm = np.dot(np.conjugate(Pcg_).T, Pcg_)
Pcg = np.sqrt(np.add(1, np.dot(Pcg_.T, Pcg_)))
Pcg = np.dot(np.dot(2, Pcg_), np.linalg.inv(Pcg))
Rcg = quatMinimal2rot(np.divide(Pcg, 2)).reshape(3, 3)

A = []
B = []
id = 0
for i in range(len(Hgs)):
    for j in range(i + 1, len(Hgs)):
        Hgij = Hgijs[id]
        Hcij = Hcijs[id]
        A.append(np.subtract(Hgij[0:3, 0:3], np.eye(3, 3)))
        B.append(np.subtract(np.dot(Rcg, Hcij[0:3, 3:4]), Hgij[0:3, 3:4]))
        id += 1

MA = np.asarray(A).reshape(size * 3, 3)
MB = np.asarray(B).reshape(size * 3, 1)
Tcg = np.dot(np.linalg.pinv(MA), MB).reshape(3, )
print(tfs.affines.compose(Tcg, np.squeeze(Rcg), [1, 1, 1]))
