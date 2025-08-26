import torch
import pymanopt
import numpy as np
from scipy import sparse

def step4_run(U, Args):
    # m-umap main algorithm

    ## Initialize GPU arrays
    device = "cuda" if Args.gpu else "cpu"

    U = torch.as_tensor(U, device=device).contiguous().requires_grad_(True)
    # A = torch.as_tensor(Args.A, device=device)
    # M = torch.as_tensor(Args.M, device=device)
    # Am = torch.as_tensor(Args.Am, device=device)
    # partition = torch.as_tensor(Args.partition, device=device)
    A = Args.A
    M = Args.M
    Am = Args.Am
    partition = Args.partition
    k = Args.k
    gamma = Args.gamma
    alpha = Args.alpha
    beta = Args.beta

    ## Precompute gradient matrices

    # Normalized degrees vector
    K_nrm = np.sqrt(gamma / A.sum()) * A.sum(1)

    # Modules and normalized modules
    N = M.sum(0, keepdims=True)
    M_nrm = M / N

    # Module adjacency and modularity matrices
    g = gamma * A.mean()     # mean(K)^2 / sum(K)
    # Bm == ((A - g) .* (~(M * M'))) * M
    #    == ((A - g) * M - ((A - g) .* (M * M')) * M
    #    -> then simplify (A - g) .* (M * M')
    Bm = (Am - M * Am) - (g * N - g * (M * N))

    Ic = [None] * k
    Bc = [None] * k
    Ac = [None] * k
    Kc_nrm = [None] * k
    for i in range(k):
        I = np.where(partition == i)[0]
        Ic[i] = I
        if Args.cache:
            Bc[i] = A[np.ix_(I, I)] - (K_nrm[I] * K_nrm[I].T)
        else:
            Ac[i] = A[np.ix_(I, I)]
            Kc_nrm[i] = K_nrm[I]

    as_sparse_tensor = lambda x: torch.sparse_csr_tensor(x.indptr, x.indices, x.data, device=device)

    # Convert to PyTorch tensors
    A = as_sparse_tensor(A)
    M = torch.as_tensor(M, device=device)
    Am = torch.as_tensor(Am, device=device)
    partition = torch.as_tensor(partition, device=device)
    K_nrm = torch.as_tensor(K_nrm, device=device)
    N = torch.as_tensor(N, device=device)
    M_nrm = torch.as_tensor(M_nrm, device=device)
    Bm = torch.as_tensor(Bm, device=device)
    Ic = [torch.as_tensor(ic, device=device) for ic in Ic]
    Bc = [torch.as_tensor(bc, device=device) for bc in Bc]
    Ac = [as_sparse_tensor(ac) for ac in Ac]
    Kc_nrm = [torch.as_tensor(kc_nrm, device=device) for kc_nrm in Kc_nrm]

    ## Run solvers

    match Args.solver:
        case "adam":
            vb = Args.verbose
            fp = {
                "head": lambda: print("%5s %24s %12s" % ("iter", "cost val", "grad. norm")) if vb else None,
                "iter": lambda t, cost, grad_norm: print("%5d %+.16e %12e" % (t, cost, grad_norm)) if vb else None,
                "stop_cost": lambda: print("Cost tolerance reached; tol = %g." % Args.tol) if vb else None,
                "stop_grad": lambda: print("Gradient norm tolerance reached; tol = %g." % Args.tol) if vb else None,
                "stop_iter": lambda: print("Max iter exceeded; maxiter = %g." % Args.maxiter) if vb else None
            }
            fp["head"]()

            optimizer = torch.optim.Adam([U], lr=Args.learnrate)
            CostHistory = np.full(Args.maxiter, np.nan)
            for t in range(Args.maxiter):
                cost = fx_cost(U, Ic, Bc, Ac, Kc_nrm, M_nrm, Bm, alpha, beta)
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                grad_norm = U.grad.norm().detach().cpu().numpy()
                with torch.no_grad():
                    U = U.div_(U.norm(dim=1, keepdim=True))

                cval = cost.detach().cpu().numpy()
                CostHistory[t] = cval
                fp["iter"](t, cval, grad_norm)
                if t and (abs(cval - CostHistory[t-1]) < Args.tol):
                    fp["stop_cost"]()
                    break
                elif grad_norm < Args.tol:
                    fp["stop_grad"]()
                    break
                elif t == Args.maxiter - 1:
                    fp["stop_iter"]()
                    break

            U = U.detach().cpu().numpy()
            CostHistory = CostHistory[~np.isnan(CostHistory)]

        case "trustregions":
            # Create the problem structure.
            manifold = pymanopt.manifolds.Oblique(Args.d, Args.n)   # transposed to normalize rows
            fx_ucost = lambda U: fx_cost(U.T, Ic, Bc, Ac, Kc_nrm, M_nrm, Bm, alpha, beta)
            problem = pymanopt.Problem(manifold, cost=pymanopt.function.pytorch(manifold)(fx_ucost))
            optimizer = pymanopt.optimizers.TrustRegions()
            result = optimizer.run(problem)

            U = result.point.detach().cpu().numpy().T               # transposed back
            CostHistory = np.array([h["cost"] for h in result.history if "cost" in h], dtype=float)

    return U, CostHistory

def fx_cost(U, Ic, Bc, Ac, Kc_nrm, M_nrm, Bm, alpha, beta):

    k = len(Ic)

    ## Compute mean-field between-module cost
    # UUm == ((U * U') .* (~(M * M'))) * Mn
    UUm = U @ (U.T @ M_nrm)
    for i in range(k):
        I = Ic[i]
        UUm[I, i] = 0          # exclude self-modules

    Dm = 2 * (1 - UUm)
    Numm = beta * alpha * (Dm ** (beta - 1))
    # Numm.fill_diagonal_(0)     # don't need / matrix is non-square
    if beta >= 1:                # fast update
        Denm = 1 + Numm * Dm / beta
    else:                        # avoid NaN
        Denm =  1 + alpha * (Dm ** beta)

    Cost = - torch.sum(Bm / Denm)

    ## Compute full within-module cost and gradient
    for i in range(k):
        if Bc[i] is not None:
            Bi = Bc[i]
        else:
            Bi = Ac[i] - (Kc_nrm[i] * Kc_nrm[i].T)

        I = Ic[i]
        Ui = U[I]
        ni = len(Ui)   # number of nodes in module i
        Di = 2 * (1 - (Ui @ Ui.T))
        Numi = beta * alpha * (Di ** (beta - 1))
        Numi.fill_diagonal_(0)
        if beta >= 1:            # fast update
            Deni = 1 + Numi * Di / beta
        else:                    # avoid NaN
            Deni =  1 + alpha * (Di ** beta)

        Cost -= torch.sum(Bi / Deni)

    return Cost

def fx_cost_full(U, B, alpha, beta):
    ## Compare full cost and gradient
    D = 2 * (1 - (U @ U.T))
    Num = beta * alpha * (D ** (beta - 1))
    Num.fill_diagonal_(0)
    Den1 =   1 + alpha * (D ** beta)
    Cost =  - torch.sum(B / Den1)

    return Cost

