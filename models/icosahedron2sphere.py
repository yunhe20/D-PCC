# https://github.com/23michael45/PanoContextTensorflow/blob/d14a3ba2bc44fcf0471fe7160fe85f077e725641/PanoContextTensorflow/icosahedron2sphere.py
import numpy as np

def icosahedron2sphere(level):
    # this function uses a icosahedron to sample uniformly on a sphere
    a = 2 / (1 + np.sqrt(5))
    M = np.array([
        0, a, -1, a, 1, 0, -a, 1, 0,
        0, a, 1, -a, 1, 0, a, 1, 0,
        0, a, 1, 0, -a, 1, -1, 0, a,
        0, a, 1, 1, 0, a, 0, -a, 1,
        0, a, -1, 0, -a, -1, 1, 0, -a,
        0, a, -1, -1, 0, -a, 0, -a, -1,
        0, -a, 1, a, -1, 0, -a, -1, 0,
        0, -a, -1, -a, -1, 0, a, -1, 0,
        -a, 1, 0, -1, 0, a, -1, 0, -a,
        -a, -1, 0, -1, 0, -a, -1, 0, a,
        a, 1, 0, 1, 0, -a, 1, 0, a,
        a, -1, 0, 1, 0, a, 1, 0, -a,
        0, a, 1, -1, 0, a, -a, 1, 0,
        0, a, 1, a, 1, 0, 1, 0, a,
        0, a, -1, -a, 1, 0, -1, 0, -a,
        0, a, -1, 1, 0, -a, a, 1, 0,
        0, -a, -1, -1, 0, -a, -a, -1, 0,
        0, -a, -1, a, -1, 0, 1, 0, -a,
        0, -a, 1, -a, -1, 0, -1, 0, a,
        0, -a, 1, 1, 0, a, a, -1, 0])

    coor = M.T.reshape(3, 60, order='F').T
    coor, idx = np.unique(coor, return_inverse=True, axis=0)
    tri = idx.reshape(3, 20, order='F').T

    # extrude
    coor = list(coor / np.tile(np.linalg.norm(coor, axis=1, keepdims=True), (1, 3)))

    for _ in range(level):
        triN = []
        for t in range(len(tri)):
            n = len(coor)
            coor.append((coor[tri[t, 0]] + coor[tri[t, 1]]) / 2)
            coor.append((coor[tri[t, 1]] + coor[tri[t, 2]]) / 2)
            coor.append((coor[tri[t, 2]] + coor[tri[t, 0]]) / 2)

            triN.append([n, tri[t, 0], n+2])
            triN.append([n, tri[t, 1], n+1])
            triN.append([n+1, tri[t, 2], n+2])
            triN.append([n, n+1, n+2])
        tri = np.array(triN)

        # uniquefy
        coor, idx = np.unique(coor, return_inverse=True, axis=0)
        tri = idx[tri]

        # extrude
        coor = list(coor / np.tile(np.sqrt(np.sum(coor * coor, 1, keepdims=True)), (1, 3)))

    return np.array(coor), np.array(tri)

