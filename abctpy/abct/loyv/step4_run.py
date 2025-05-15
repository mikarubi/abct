from types import SimpleNamespace
import warnings

import numpy as np

def step4_run(Args, W, M, My):
    # Loyvain main algorithm

    Args = SimpleNamespace(**Args)

    # Unpack arguments
    k = Args.k

    n = len(M)
    LinIdx = M + k * np.arange(n)  # two-dimensional indices of M

    MM = np.zeros((k, n))
    MM[M, np.arange(n)] = 1  # two-dimensional representation
    N = np.sum(MM, axis=1)  # number of nodes in module
    match Args.method:
        case "loyvain":
            if Args.similarity == "network":
                Smn = MM @ W  # degree of module to node
            else:
                X = Args.X
                G = MM @ X  # cluster centroid
                Smn = G @ X.T  # dot of centroid with node
            Cii = np.diag(Smn @ MM.T)  # within-module weight sum
            if Args.objective == "spectral":
                S = np.sum(Smn, axis=0)  # degree of node
                D = np.sum(Smn, axis=1)  # degree of module
            Wii = Args.Wii  # within-node weight sum

        case "coloyvain":
            ny = len(My)
            MMy = np.zeros((k, ny))
            MMy[My, np.arange(ny)] = 1  # two-dimensional representation
            Ny = np.sum(MMy, axis=1)
            Smn = MMy @ W.T  # strength node to module of Wxy
            Cii = np.diag(MM @ Smn.T)  # within-module weight sum
            if Args.objective == "cospectral":
                S = np.sum(W, axis=1)
                D = np.sum(MM @ W, axis=1)
                Dy = np.sum(MMy @ W.T, axis=1)

    match Args.objective:
        case "kmeans":
            Cii_nrm = Cii / N
        case "spectral":
            Cii_nrm = Cii / D
        case "cokmeans":
            Cii_nrm = Cii / np.sqrt(N * Ny)
        case "cospectral":
            Cii_nrm = Cii / np.sqrt(D * Dy)

    if (k == 1) or (k == n):
        Args.maxiter = 0  # skip loop if trivial partition

    for v in range(Args.maxiter):

        max_delta_Q = 0  # maximal increase over all batches
        idx = np.random.permutation(n)
        pts = np.round(np.linspace(0, n, Args.numbatches + 1)).astype(int)
        Batches = [idx[pts[i] : pts[i + 1]] for i in range(len(pts) - 1)]
        for u in range(Args.numbatches):
            U = Batches[u]  # indices of nodes in batch
            LinU = LinIdx[U]  # linear indices of nodes in batch
            MU = M[U]  # module assignments of nodes in batch
            b = len(U)  # number of nodes in batch

            match Args.objective:
                case "kmeans":
                    delta_QU = ((2 * Smn[:, U] + Wii[U]) - Cii_nrm) / (N + 1) - ((2 * Smn[LinU] - Wii[U]) - Cii_nrm[MU]) / (N[MU] - 1)
                case "spectral":
                    delta_QU = ((2 * Smn[:, U] + Wii[U]) - Cii_nrm * S[U]) / (D + S[U]) - ((2 * Smn[LinU] - Wii[U]) - Cii_nrm[MU] * S[U]) / (D[MU] - S[U])
                case "cokmeans":
                    delta_QU = ((Cii + Smn[:, U]) / np.sqrt((N + 1) * Ny) - Cii_nrm + (Cii[MU] - Smn[LinU]) / np.sqrt((N[MU] - 1) * Ny[MU]) - Cii_nrm[MU])
                case "cospectral":
                    delta_QU = ((Cii + Smn[:, U]) / np.sqrt((D + S[U]) * Dy) - Cii_nrm + (Cii[MU] - Smn[LinU]) / np.sqrt((D[MU] - S[U]) * Dy[MU]) - Cii_nrm[MU])

            delta_QU[:, N[MU] == 1] = -np.inf  # no change allowed if one-node cluster
            delta_QU[MU + k * np.arange(b)] = 0  # no change if node stays in own module

            # Update if improvements
            max_delta_QU = np.max(delta_QU, axis=0)
            MU_new = np.argmax(delta_QU, axis=0)
            if np.max(max_delta_QU) > Args.tolerance:
                max_delta_Q = np.maximum(max_delta_Q, np.max(max_delta_QU))

                IU = np.where(MU != MU_new)[0]  # batch indices of nodes to be switched
                I = U[IU]  # actual indices of nodes to be switched
                MI_new = MU_new[IU]  # new module assignments

                # get delta modules and ensure non-empty modules
                n_i = len(I)
                MMI = np.zeros((k, n_i))
                MMI[M[I], np.arange(n_i)] = 1
                while True:
                    MMI_new = np.zeros((k, n_i))
                    MMI_new[MI_new, np.arange(n_i)] = 1
                    delta_MMI = MMI_new - MMI
                    N_new = N + np.sum(delta_MMI, axis=1)
                    if np.all(N_new):
                        break
                    else:
                        E = np.where(N_new == 0)[0]  # empty modules
                        k_e = len(E)  # number of empty modules
                        MI_new[np.random.permutation(n_i)[:k_e]] = E

                # Update all relevant variables
                N = N_new
                M[I] = MI_new
                MM = np.zeros((k, n))
                MM[M, np.arange(n)] = 1
                LinIdx[I] = MI_new + k * (I - 1)

                if Args.method == "loyvain":
                    # Update G and Smn
                    if Args.similarity == "network":
                        delta_Smn = delta_MMI @ W[I, :]
                    else:
                        delta_G = delta_MMI @ X[I, :]  # change in centroid
                        G = G + delta_G  # update centroids
                        delta_Smn = delta_G @ X.T  # change in degree of module to node
                    Smn = Smn + delta_Smn  # update degree of module to node
                    Cii = np.diag(Smn @ MM.T)  # within-module weight sum
                elif Args.method == "coloyvain":
                    Cii = np.diag(MM @ Smn.T)  # within-module weight sum
                    if Args.objective == "cospectral":
                        delta_Smn = delta_MMI @ W[I, :]

                if Args.objective in ["spectral", "cospectral"]:
                    D = D + np.sum(delta_Smn, axis=1)

                match Args.objective:
                    case "kmeans":
                        Cii_nrm = Cii / N
                    case "spectral":
                        Cii_nrm = Cii / D
                    case "cokmeans":
                        Cii_nrm = Cii / np.sqrt(N * Ny)
                    case "cospectral":
                        Cii_nrm = Cii / np.sqrt(D * Dy)

        if max_delta_Q < Args.tolerance:
            break

        if (Args.display == "iteration") and (Args.method == "loyvain"):
            print(
                f"Replicate: {Args.replicate_i:4d}.    Iteration: {v:4d}.    Largest Δ: {max_delta_Q:4.4f}"
            )
        if v == Args.maxiter:
            warnings.warn(f"Algorithm did not converge after {v} iterations.")

    # Return objective
    Q = np.sum(Cii_nrm)

    return M, Q, Cii_nrm
