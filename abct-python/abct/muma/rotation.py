from numpy.typing import ArrayLike

import numpy as np
from abct import muma

def rotation(U3: ArrayLike) -> tuple[np.ndarray, np.ndarray]:

    # ensure spherical geometry
    n = len(U3)
    U3 = U3 / np.linalg.norm(U3, axis=1, keepdims=True)

    ## Rotate data to empty poles

    p = 144                     # Fibonacci number
    V = muma.fsphere(p)         # Unit Fibonacci sphere

    # Poles have minimal (maximal correlations to all other points)
    D = np.max(np.abs(U3 @ V.T), axis=0)
    idx = np.argmin(D)

    # Rotate data so that v becomes [0, 0, 1]
    v = V[idx]
    Rp = rota(v, 2)
    U3 = U3 @ Rp

    ## Rotate data to sparse anti-prime meridian

    # Unit circle
    Theta = np.pi * np.arange(-p, p, 2)/p
    V = np.column_stack((np.cos(Theta), np.sin(Theta)))

    # Sparse meridians have few strong correlations to other points
    alpha = 12*np.pi/p;         # degree band (6/p in either direction)
    D = np.zeros(p)
    for i in range(p):
        Vi = np.hstack((V[np.full(n, i)], U3[:, 2:3]))
        Vi = Vi / np.linalg.norm(Vi, axis=1, keepdims=True)
        # Number of points within the angle boundary
        D[i] = np.mean(np.sum(U3 * Vi, axis=1) > np.cos(alpha))

    # Find middle meridian among several identical
    Comps = np.hstack((D, D)) == np.min(D)         # Enforce circular boundary conditions
    Comps = np.hstack((0, Comps, 0))
    Comps = np.diff(Comps)
    Comp_sta = np.where(Comps ==  1)[0]
    Comp_fin = np.where(Comps == -1)[0]
    u = np.argmax(Comp_fin - Comp_sta)
    idx = (Comp_sta[u] + Comp_fin[u]) // 2
    v = np.hstack((V[idx % p], 0))

    # Rotate U3 so that v is placed on [-1, 0, 0]
    Rm = rota(-v, 0)
    U3 = U3 @ Rm

    # Combine rotations
    R = Rp @ Rm

    return U3, R

def rota(v: ArrayLike, a: int) -> np.ndarray:
    # Rotate data such that v aligns with a basis vector
    epss = np.finfo(np.float32).eps

    e = np.zeros(3)
    e[a] = 1
    b = np.where(e == 0)[0][0]
    if np.linalg.norm(np.abs(e) - np.abs(v)) < epss:    # Target vector parallel
        R = np.eye(3)
        if np.linalg.norm(e - (- v)) < epss:            # Target vector antiparallel
            R[a, a] = -1
            R[b, b] = -1
    else:
        # Cross product
        x = np.cross(e, v)
        cos_theta = np.dot(e, v)
        sin_theta = np.linalg.norm(x)

        # Skew-symmetric matrix
        Q = np.array([[0, -x[2], x[1]],
                      [x[2], 0, -x[0]],
                      [-x[1], x[0], 0]])

        # Rodrigues formula
        R = np.eye(3) + Q + Q @ Q * ((1 - cos_theta)/(sin_theta**2))

    # Check that rotation matrix is valid and rotation is correct
    assert((np.linalg.norm(np.eye(3) - R.T @ R) < epss) and (np.abs(1 - np.linalg.det(R)) < epss))
    assert(np.linalg.norm(e - v @ R) < epss)

    return R
