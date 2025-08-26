function [U, CostHistory] = step4_run(U, Args)
% m-umap main algorithm

k = Args.k;
gamma = Args.gamma;
alpha = Args.alpha;
beta = Args.beta;
A = Args.A;
M = Args.M;
Am = Args.Am;

%% Precompute gradient matrices

% Normalized degrees vector
K_nrm = sqrt(gamma / full(sum(A, "all"))) * full(sum(A, 2));

% Modules and normalized modules
N = sum(M);
M_nrm = M./N;

% Module adjacency and modularity matrices
g = gamma * mean(A, "all");     % mean(K)^2 / sum(K);
% Bm == ((A - g) .* (~(M * M'))) * M
%    == ((A - g) * M - ((A - g) .* (M * M')) * M
%    -> then simplify (A - g) .* (M * M')
Bm = full((Am - M .* Am) - (g * N - g .* (M .* N)));

Ic = cell(k, 1);
Bc = cell(k, 1);
Ac = cell(k, 1);
Kc_nrm = cell(k, 1);
for i = 1:k
    I = find(Args.partition == i);
    Ic{i} = I;
    if Args.cache
        Bc{i} = full(A(I, I)) - (K_nrm(I) * K_nrm(I)');
    else
        Ac{i} = A(I, I);
        Kc_nrm{i} = K_nrm(I);
    end
end

%% Initialize GPU arrays

if Args.gpu
    for i = 1:k
        Ic{i} = gpuArray(Ic{i});
        Bc{i} = gpuArray(Bc{i});
        Ac{i} = gpuArray(Ac{i});
        Kc_nrm{i} = gpuArray(Kc_nrm{i});
    end
    U = gpuArray(U);
    M_nrm = gpuArray(M_nrm);
    Bm = gpuArray(Bm);
    alpha = gpuArray(alpha);
    beta = gpuArray(beta);
end

%% Run solvers

switch Args.solver
    case "adam"
        if Args.verbose
            fp.head = @() fprintf("%5s %24s %12s\n", "iter", "cost val", "grad. norm");
            fp.iter = @(t, cost, grad_norm) fprintf("%5d %+.16e %12e\n", t, cost, grad_norm);
            fp.stop_cost = @() fprintf("Cost tolerance reached; tol = %g.\n", Args.tol);
            fp.stop_grad = @() fprintf("Gradient norm tolerance reached; tol = %g.\n", Args.tol);
            fp.stop_iter = @() fprintf("Max iter exceeded; maxiter = %g.", Args.maxiter);
        else
            fp = struct(head = @()[], iter = @(a,b,c)[], stop_cost = @()[], stop_grad = @()[], stop_iter = @()[]);
        end
        fp.head();

        ave_grad  = zeros(size(U));   % 1st-moment estimates
        ave_grad2 = zeros(size(U));   % 2nd-moment estimates
        CostHistory = nan(1, Args.maxiter);
        for t = 1:Args.maxiter
            [cost, grad] = costgrad(U, Ic, Bc, Ac, Kc_nrm, M_nrm, Bm, alpha, beta);

            [U, ave_grad, ave_grad2] = adamupdate(U, grad, ave_grad, ave_grad2, t, Args.learnrate);
            grad_norm = norm(grad, "fro");
            U = U ./ vecnorm(U,2,2);
            CostHistory(t) = cost;

            fp.iter(t, cost, grad_norm);
            if (t > 1) && (abs(cost - CostHistory(t-1)) < Args.tol)
                fp.stop_cost(); break;
            elseif grad_norm < Args.tol
                fp.stop_grad(); break;
            elseif t == Args.maxiter
                fp.stop_iter(); break;
            end
        end
        CostHistory = CostHistory(~isnan(CostHistory));

    case "trustregions"
        % Create the problem structure.
        problem.M = obliquefactory(Args.n, Args.d, "rows", Args.gpu);

        % Get the modularity matrix
        problem.costgrad = @(U) costgrad(U, Ic, Bc, Ac, Kc_nrm, M_nrm, Bm, alpha, beta);
        % checkgradient(problem);

        opts = struct(tolgradnorm=Args.tol, maxiter=Args.maxiter, verbosity=2*Args.verbose);
        [U, ~, info] = trustregions(problem, U, opts);
        CostHistory = info.cost;
end

% Gather from GPU
if Args.gpu
    U = gather(U);
end

end

function [Cost, RGrad] = costgrad(U, Ic, Bc, Ac, Kc_nrm, M_nrm, Bm, alpha, beta)

k = length(Ic);

%% Compute mean-field between-module cost and gradient

% UUm == ((U * U') .* (~(M * M'))) * Mn
UUm = U * (U' * M_nrm);
for i = 1:k
    I = Ic{i};
    UUm(I, i) = 0;          % exclude self-modules
end

Dm = 2 * (1 - UUm);
Numm = beta * alpha * (Dm.^(beta - 1));
% Numm(1:k+1:end) = 0;      % don't need / matrix is non-square
if beta >= 1                % fast update
    Denm = 1 + Numm .* Dm / beta;
else                        % avoid NaN
    Denm =  1 + alpha * (Dm.^(beta));
end
Cost = - sum(Bm ./ Denm, "all");

G = - 2 * Bm .* Numm ./ (Denm.^2);   % n x k
EGrad = G * (M_nrm' * U) + M_nrm * (G' * U);

%% Compute full within-module cost and gradient

for i = 1:k
    if ~isempty(Bc{i})
        Bi = Bc{i};
    else
        Bi = full(Ac{i}) - (Kc_nrm{i} * Kc_nrm{i}');
    end

    I = Ic{i};
    Ui = U(I, :);
    ni = size(Ui, 1);   % number of nodes in module i
    Di = 2 * (1 - (Ui * Ui'));
    Numi = beta * alpha * (Di.^(beta - 1));
    Numi(1:ni+1:end) = 0;
    if beta >= 1            % fast update
        Deni = 1 + Numi .* Di / beta;
    else                    % avoid NaN
        Deni =  1 + alpha * (Di.^(beta));
    end
    Cost = Cost - sum(Bi ./ Deni, "all");
    EGrad(I, :) = EGrad(I, :) - (4 * Bi .* Numi ./ (Deni.^2)) * Ui;
end

% Orthogonal projection of H in R^(nxm) to the tangent space at X.
% Compute the inner product between each column/row of H with the
% corresponding column/row of X. Remove from H the components that are
% parallel to X, by row/col.
U_dot_EGrad = sum(U .* EGrad, 2);
RGrad = EGrad - U .* U_dot_EGrad;

end

function [Cost, RGrad] = costgrad_full(U, B, alpha, beta)
%% Compare full cost and gradient
n = size(U, 1);
D = 2 * (1 - (U * U'));
Num = beta * alpha * (D.^(beta - 1));
Num(1:n+1:end) = 0;
Den1 =   1 + alpha * (D.^(beta));
Cost =  - sum(B ./ Den1, "all");
EGrad = - (4 * B .* Num ./ (Den1.^2)) * U;
U_dot_EGrad = sum(U .* EGrad, 2);
RGrad = EGrad - U .* U_dot_EGrad;

end
