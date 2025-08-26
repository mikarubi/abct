import numpy as np
from scipy import sparse
from abct import muma

def step3_init(Args):
    # m-umap output initialization

    A = Args.A
    M = Args.M
    Am = Args.Am
    k = Args.k
    match Args.start:
        case "custom":
            U = Args.U
            U = U / np.linalg.norm(U, axis=1, keepdims=True)
        case "spectral_nn":                         # spectral on knn matrix
            _, U = sparse.linalg.eigs(A, Args.d+1)
            U = U[:, 1:]
            U = U / np.linalg.norm(U, axis=1, keepdims=True)
        case "spectral":                            # spectral on modules
            Amm = M.T @ Am
            Kmm = np.sum(Amm, axis=1, keepdims=True)
            Bmm = Amm - Kmm * Kmm.T / np.sum(Amm)
            _, Um = sparse.linalg.eigs(Bmm, Args.d)
            Um = Um / np.linalg.norm(Um, axis=1, keepdims=True)
            U = Um[Args.partition]
        case "greedy":                              # spherical maximin
            Amm = M.T @ Am                          # module connectivity
            np.fill_diagonal(Amm, np.nan)           # ignore self-connections
            Kmm_ = np.zeros(k)                      # degree to placed modules
            Um = np.zeros((k, Args.d))              # module locations
            Vm = muma.fsphere(k)                    # Fibonacci sphere
            ux = np.argmax(np.nansum(Amm, axis=1))  # initial module index
            vx = np.argmax(np.nansum(Vm @ Vm.T, axis=1)) # initial location index
            for i in range(k):
                Um[ux] = Vm[vx]                     # assign location
                Vm[vx] = np.nan                     # remove point from consideration
                if i == k-1:
                    break
                Kmm_ = Kmm_ + Amm[:, ux]            # add module connectivity (with self-nan's)
                ux = np.nanargmin(Kmm_)             # least connected module (nan's in Kmm mask set modules)
                vx = np.nanargmin(Vm @ np.mean(Um, axis=0, keepdims=True).T)  # furthest location (nan's in Vm mask used locations)
            U = Um[Args.partition]

    return U
