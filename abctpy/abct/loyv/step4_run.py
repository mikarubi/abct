import warnings

import numpy as np

def step4_run(Args, W, M, My=None):
    # Loyvain main algorithm

    # Unpack arguments
    k = Args.k
    n = len(M)

    MM = np.zeros((k, n))
    MM[M, np.arange(n)] = 1  # two-dimensional representation
    N = np.sum(MM, axis=1, keepdims=True)  # number of nodes in module
    match Args.method:
        case "loyvain":
            if Args.similarity == "network":
                Smn = MM @ W  # degree of module to node
            else:
                X = Args.X
                G = MM @ X  # cluster centroid
                Smn = G @ X.T  # dot of centroid with node
            Cii = np.diag(Smn @ MM.T)[:, np.newaxis]  # within-module weight sum
            if Args.objective == "spectral":
                S = np.sum(Smn, axis=0, keepdims=True)  # degree of node
                D = np.sum(Smn, axis=1, keepdims=True)  # degree of module
            Wii = Args.Wii[np.newaxis, :]  # within-node weight sum

        case "coloyvain":
            ny = len(My)
            MMy = np.zeros((k, ny))
            MMy[My, np.arange(ny)] = 1  # two-dimensional representation
            Ny = np.sum(MMy, axis=1, keepdims=True)  # number of nodes in module
            Smn = MMy @ W.T  # strength node to module of Wxy
            Cii = np.diag(MM @ Smn.T)[:, np.newaxis]  # within-module weight sum
            if Args.objective == "cospectral":
                S = np.sum(W, axis=1, keepdims=True)
                D = np.sum(MM @ W, axis=1, keepdims=True)
                Dy = np.sum(MMy @ W.T, axis=1, keepdims=True)

    match Args.objective:
        case "kmeans":      Cii_nrm = Cii / N
        case "spectral":    Cii_nrm = Cii / D
        case "cokmeans":    Cii_nrm = Cii / np.sqrt(N * Ny)
        case "cospectral":  Cii_nrm = Cii / np.sqrt(D * Dy)

    if (k == 1) or (k == n):
        Args.maxiter = 0  # skip loop if trivial partition

    for v in range(Args.maxiter):
        max_delta_Q = 0  # maximal increase over all batches
        idx = np.random.permutation(n)
        pts = np.round(np.linspace(0, n, Args.numbatches + 1)).astype(int)
        Batches = [np.array(idx[pts[i]:pts[i+1]]) for i in range(len(pts) - 1)]
        for u in range(Args.numbatches):
            U = Batches[u]  # indices of nodes in batch
            MU = M[U]  # module assignments of nodes in batch

            # Calculate change in modularity
            with np.errstate(divide="ignore", invalid="ignore"):
                match Args.objective:
                    case "kmeans":
                        delta_QU = (((2 * Smn[:, U] + Wii[:, U]) - Cii_nrm) / (N + 1) -
                                ((2 * Smn[MU, U] - Wii[:, U]) - Cii_nrm[MU].T) / (N[MU].T - 1))
                    case "spectral":
                        delta_QU = (((2 * Smn[:, U] + Wii[:, U]) - Cii_nrm * S[:, U]) / (D + S[:, U]) -
                                    ((2 * Smn[MU, U] - Wii[:, U]) - Cii_nrm[MU].T * S[:, U]) / (D[MU].T - S[:, U]))
                    case "cokmeans":
                        delta_QU = ((Cii + Smn[:, U]) / np.sqrt((N + 1) * Ny) - Cii_nrm + 
                                    (Cii[MU].T - Smn[MU, U]) / np.sqrt((N[MU].T - 1) * Ny[MU].T) - Cii_nrm[MU].T)
                    case "cospectral":
                        delta_QU = ((Cii + Smn[:, U]) / np.sqrt((D + S[:, U]) * Dy) - Cii_nrm + 
                                    (Cii[MU].T - Smn[MU, U]) / np.sqrt((D[MU].T - S[:, U]) * Dy[MU].T) - Cii_nrm[MU].T)

            delta_QU[:, (N[MU] == 1).ravel()] = -np.inf  # no change allowed if one-node cluster
            delta_QU[MU, np.arange(len(U))] = 0  # no change if node stays in own module

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
                while True:
                    MMI_new = np.zeros((k, n_i))
                    MMI_new[MI_new, np.arange(n_i)] = 1
                    delta_MMI = MMI_new - MM[:, I]
                    N_new = N + np.sum(delta_MMI, axis=1, keepdims=True)
                    if np.all(N_new):
                        break
                    else:
                        E = np.where(N_new == 0)[0]  # empty modules
                        k_e = len(E)  # number of empty modules
                        MI_new[np.random.choice(n_i, k_e, replace=False)] = E

                # Update all relevant variables
                N = N_new
                M[I] = MI_new
                MM[:, I] = MMI_new

                match Args.method:
                    case "loyvain":
                        # Update G and Smn
                        if Args.similarity == "network":
                            delta_Smn = delta_MMI @ W[I]
                        else:
                            delta_G = delta_MMI @ X[I]  # change in centroid
                            G = G + delta_G  # update centroids
                            delta_Smn = delta_G @ X.T  # change in degree of module to node
                        Smn = Smn + delta_Smn  # update degree of module to node
                        Cii = np.diag(Smn @ MM.T)[:, np.newaxis]  # within-module weight sum
                    case "coloyvain":
                        Cii = np.diag(MM @ Smn.T)[:, np.newaxis]  # within-module weight sum
                        if Args.objective == "cospectral":
                            delta_Smn = delta_MMI @ W[I]

                if Args.objective in ["spectral", "cospectral"]:
                    D = D + np.sum(delta_Smn, axis=1, keepdims=True)

                match Args.objective:
                    case "kmeans":      Cii_nrm = Cii / N
                    case "spectral":    Cii_nrm = Cii / D
                    case "cokmeans":    Cii_nrm = Cii / np.sqrt(N * Ny)
                    case "cospectral":  Cii_nrm = Cii / np.sqrt(D * Dy)

        if max_delta_Q < Args.tolerance:
            break

        if (Args.display == "iteration") and (Args.method == "loyvain"):
            print(f"Replicate: {Args.replicate_i:4d}.    Iteration: {v:4d}.    Largest Δ: {max_delta_Q:4.4f}")
        if v == Args.maxiter:
            warnings.warn(f"Algorithm did not converge after {v} iterations.")

    # Return objective
    Q = np.sum(Cii_nrm)

    return M, Q, Cii_nrm
