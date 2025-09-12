import torch
import pymanopt
import numpy as np
from scipy import sparse
import warnings

def step4_run(U, Args):
    # m-umap main algorithm

    ## Initialize GPU arrays
    device = "cuda" if Args.gpu else "cpu"

    k = Args.k
    gamma = Args.gamma
    alpha = Args.alpha
    beta = Args.beta
    A = Args.A
    M = Args.M
    Am = Args.Am

    ## Precompute gradient matrices

    # Normalized degree column vector
    K_nrm = np.sqrt(gamma / A.sum()) * A.sum(1)[:, np.newaxis]

    # Modules and normalized modules
    N = M.sum(0, keepdims=True)
    M_nrm = M / N

    # Module adjacency and modularity matrices
    g = gamma * A.mean()     # mean(K)^2 / sum(K)
    # Bm == ((A - g) .* (~(M * M'))) * M
    #    == ((A - g) * M - ((A - g) .* (M * M')) * M
    #    -> then simplify (A - g) .* (M * M')
    Bm = (Am - M * Am) - (g * N - g * (M * N))

    Ic = [np.array([])] * k
    Bc = [np.array([])] * k
    Ac = [sparse.csr_array([])] * k
    Kc_nrm = [np.array([])] * k
    for i in range(k):
        I = np.where(Args.partition == i)[0]
        Ic[i] = I
        if Args.cache:
            Bc[i] = A[np.ix_(I, I)] - (K_nrm[I] * K_nrm[I].T)
        else:
            Ac[i] = A[np.ix_(I, I)]
            Kc_nrm[i] = K_nrm[I]

    ## Initialize Torch tensors

    U = torch.as_tensor(U, device=device).contiguous().requires_grad_(True)
    M_nrm = torch.as_tensor(M_nrm, device=device)
    Bm = torch.as_tensor(Bm, device=device)
    alpha = torch.as_tensor(alpha, device=device)
    beta = torch.as_tensor(beta, device=device)
    for i in range(k):
        Ic[i] = torch.as_tensor(Ic[i], device=device)
        Bc[i] = torch.as_tensor(Bc[i], device=device)
        with warnings.catch_warnings(action="ignore"):
            Ac[i] = torch.sparse_csr_tensor(Ac[i].indptr, Ac[i].indices, Ac[i].data, device=device)
        Kc_nrm[i] = torch.as_tensor(Kc_nrm[i], device=device)

    ## Run solvers

    match Args.solver:
        case "adam":
            vb = Args.verbose
            fp = {
                "head": lambda: print("%5s %24s %12s" % ("iter", "cost val", "grad. norm")) if vb else None,
                "iter": lambda t, cost, grad_norm: print("%5d %+.16e %12e" % (t, cost, grad_norm)) if vb else None,
                "stop_cost": lambda: print("Cost tolerance reached; tolerance = %g." % Args.tolerance) if vb else None,
                "stop_grad": lambda: print("Gradient norm tolerance reached; tolerance = %g." % Args.tolerance) if vb else None,
                "stop_iter": lambda: print("Max iter exceeded; maxiter = %g." % Args.maxiter) if vb else None
            }
            fp["head"]()

            optimizer = torch.optim.Adam([U], lr=Args.learnrate)
            CostHistory = np.full(Args.maxiter, np.nan)
            for t in range(Args.maxiter):
                cost = fx_cost(U, Ic, Bc, Ac, Kc_nrm, M_nrm, Bm, alpha, beta)
                optimizer.zero_grad()
                cost.backward()
                with torch.no_grad():  # Get Riemannian gradient
                    U_dot_EGrad = (U * U.grad).sum(dim=1, keepdim=True)
                    U.grad -= U * U_dot_EGrad
                optimizer.step()

                grad_norm = U.grad.norm().detach().cpu().numpy()
                with torch.no_grad():
                    U /= U.norm(dim=1, keepdim=True)

                cval = cost.detach().cpu().numpy()
                CostHistory[t] = cval
                fp["iter"](t, cval, grad_norm)
                if t and (abs(cval - CostHistory[t-1]) < Args.tolerance):
                    fp["stop_cost"]()
                    break
                elif grad_norm < Args.tolerance:
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
            optimizer = pymanopt.optimizers.TrustRegions(max_time=np.inf)
            result = optimizer.run(problem)

            U = result.point.T                                      # transposed back
            CostHistory = result.cost

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
    if beta >= 1:                # fast update
        Denm = 1 + Numm * Dm / beta
    else:                        # avoid NaN
        Denm =  1 + alpha * (Dm ** beta)

    Cost = - torch.sum(Bm / Denm)

    ## Compute full within-module cost and gradient
    for i in range(k):
        if len(Bc[i]):           # Args.cache is True
            Bi = Bc[i]
        else:
            Bi = Ac[i].to_dense() - (Kc_nrm[i] * Kc_nrm[i].T)

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

