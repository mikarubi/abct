import torch
import pymanopt
import numpy as np

def step4_run(U, Args):
    # m-umap main algorithm

    ## Initialize GPU arrays
    if Args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    U = torch.as_tensor(U, device=device).contiguous().requires_grad_(True)
    A = torch.as_tensor(Args.A, device=device)
    M = torch.as_tensor(Args.M, device=device)
    Am = torch.as_tensor(Args.Am, device=device)
    k = Args.k
    gamma = Args.gamma
    alpha = Args.alpha
    beta = Args.beta

    ## Precompute gradient matrices

    # Normalized degrees vector
    K_nrm = torch.sqrt(gamma / A.sum()) * A.sum(1)

    # Modules and normalized modules
    N = M.sum()
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
        I = torch.where(Args.partition == i)[0]
        Ic[i] = I
        if Args.cache:
            Bc[i] = A[torch.ix_(I, I)] - (K_nrm[I] * K_nrm[I].T)
        else:
            Ac[i] = A[torch.ix_(I, I)]
            Kc_nrm[i] = K_nrm[I]

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

                grad_norm = U.norm()
                with torch.no_grad():
                    U = U / U.norm(1, keepdim=True)

                CostHistory[t] = cost
                fp["iter"](t, cost, grad_norm)
                if t and (abs(cost - CostHistory[t-1]) < Args.tol):
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
            manifold = pymanopt.manifolds.Oblique(Args.d, Args.n)
            fx_ucost = lambda U: fx_cost(U, Ic, Bc, Ac, Kc_nrm, M_nrm, Bm, alpha, beta)
            problem = pymanopt.Problem(manifold, cost=pymanopt.function.pytorch(manifold)(fx_ucost))
            optimizer = pymanopt.optimizers.TrustRegions()
            result = optimizer.run(problem)

            U = result.point.detach().cpu().numpy()
            CostHistory = result.history.detach().cpu().numpy()

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
    np.fill_diagonal(Numm, 0)
    if beta >= 1:                # fast update
        Denm = 1 + Numm * Dm / beta
    else:                        # avoid NaN
        Denm =  1 + alpha * (Dm ** beta)

    Cost = - np.sum(Bm / Denm)

    ## Compute full within-module cost and gradient
    for i in range(k):
        if Bc[i]:
            Bi = Bc[i]
        else:
            Bi = Ac[i] - (Kc_nrm[i] * Kc_nrm[i].T)

        I = Ic[i]
        Ui = U[I]
        ni = len(Ui)   # number of nodes in module i
        Di = 2 * (1 - (Ui @ Ui.T))
        Numi = beta * alpha * (Di ** (beta - 1))
        np.fill_diagonal(Numi, 0)
        if beta >= 1:            # fast update
            Deni = 1 + Numi * Di / beta
        else:                    # avoid NaN
            Deni =  1 + alpha * (Di ** beta)

        Cost -= np.sum(Bi / Deni)

    return Cost

def fx_cost_full(U, B, alpha, beta):
    ## Compare full cost and gradient
    D = 2 * (1 - (U @ U.T))
    Num = beta * alpha * (D ** (beta - 1))
    np.fill_diagonal(Num, 0)
    Den1 =   1 + alpha * (D ** beta)
    Cost =  - np.sum(B / Den1)

    return Cost

