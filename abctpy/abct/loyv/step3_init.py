from types import SimpleNamespace

import numpy as np


def step3_init(X, normX, Dist, n, Args):
    # Loyvain partition initialization

    Args = SimpleNamespace(**Args)

    # Unpack arguments
    k = Args.k

    if Args.start in ["greedy", "balanced"]:
        Idx = np.concatenate(np.random.randint(n), np.full(k - 1, np.nan))
        minDist = np.full(n, np.inf)
        for j in range(1, k):
            if (Args.similarity == "network") or (Args.method == "coloyvain"):
                # use precomputed distance
                Dj = Dist[Idx[j - 1], :]
            else:
                # compute distance on the fly
                Dj = 1 - (X[Idx[j - 1], :] @ X.T) / (normX[Idx[j - 1]] * normX)
            minDist = np.minimum(minDist, Dj)  # min distance to centroid
            if Args.start == "greedy":
                sampleProbability = minDist == np.max(minDist)
            elif Args.start == "balanced":
                sampleProbability = minDist / np.sum(minDist)
            P = np.concatenate(([0], np.cumsum(sampleProbability)))
            P[-1] = 1
            Idx[j] = np.searchsorted(P, np.random.rand(), side="right") - 1

        if (Args.similarity == "network") or (Args.method == "coloyvain"):
            # use precomputed distance
            M0 = np.argmin(Dist[Idx, :], axis=0)
        else:
            # compute distance on the fly
            M0 = np.argmin(
                1 - (X[Idx, :] @ X.T) / (normX[Idx][:, np.newaxis] * normX), axis=0
            )
        k0 = np.max(M0)
        if k0 < k:
            M0[np.random.choice(n, k - k0, replace=False)] = np.arange(k0 + 1, k + 1)
    elif Args.start == "random":
        M0 = np.random.randint(1, k + 1, size=n)  # initial module partition
        M0[np.random.choice(n, k, replace=False)] = np.arange(
            1, k + 1
        )  # ensure there are k modules

    return M0
