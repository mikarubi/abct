function [U, CostHistory, Args] = mumap(X, U, Args)
% MUMAP

arguments
    X (:, :) {mustBeFinite, mustBeReal}
    U (:, :) {mustBeFinite, mustBeReal} = []
    Args.d (1, 1) {mustBePositive, mustBeInteger} = 3
    Args.kappa (1, 1) {mustBePositive, mustBeInteger} = 10
    Args.alpha (1, 1) {mustBePositive} = 1
    Args.beta (1, 1) {mustBePositive} = 1/2
    Args.gamma (1, 1) {mustBePositive} = 1
    Args.Solver (1, 1) string {mustBeMember(...
        Args.Solver, ["adam", "trustregions"])} = "trustregions"
    Args.Partition (:, 1) {mustBePositive, mustBeInteger} = []
    Args.MaxIter (1, 1) {mustBePositive, mustBeInteger} = 1e4
    Args.LearnRate (1, 1) {mustBePositive} = 1e-3
    Args.Tol (1, 1) {mustBePositive} = 1e-6
    Args.GPU (1, 1) logical = false
    Args.Cache (1, 1) logical = false
    Args.Verbose (1, 1) logical = true
end

if Args.GPU
    assert(license("test", "distrib_computing_toolbox") && gpuDeviceCount, ...
        "GPU use requires Parallel Computing Toolbox and active GPU.")
end
switch Args.Solver
    case "adam"
        assert(license("test", "neural_network_toolbox"), ...
            "Adam solver requires Deep Learning Toolbox.")
    case "trustregions"
        assert((exist("obliquefactory", "file") && exist("trustregions", "file")), ...
            "Manopt functions not found. Call importmanopt to setup.")
end

%%

n = size(X, 1);
if issparse(X)
    A = X;
else
    % Generate a common neighbors matrix
    Col = knnsearch(X, X, K=Args.kappa+1, Distance="correlation");
    Col = reshape(Col, [], 1);
    Row = (1:n).';
    Row = reshape(Row(:, ones(1,Args.kappa+1)), [], 1);
    A = sparse([Row; Col], [Col; Row], true);
end
A(1:n+1:end) = 0;

% Module structure
if isempty(Args.Partition)
    [Args.Partition, Args.Q] = louvains(A, gamma=Args.gamma);
end

%% Precompute gradient matrices

% Normalized degrees vector
K_nrm = sqrt(Args.gamma / full(sum(A, "all"))) * full(sum(A, 2));

% Modules and normalized modules
M = sparse(1:n, Args.Partition, 1);
N = sum(M);
M_nrm = M./N;

% Module adjacency and modularity matrices
g = Args.gamma * mean(A, "all");     % mean(K)^2 / sum(K);
Am = A * M;
% Bm == ((A - g) .* (~(M * M'))) * M
%    == ((A - g) * M - ((A - g) .* (M * M')) * M
%    -> then simplify (A - g) .* (M * M')
Bm = full((Am - M .* Am) - (g * N - g .* (M .* N)));

k = max(Args.Partition);
Ic = cell(k, 1);
Bc = cell(k, 1);
Ac = cell(k, 1);
Kc_nrm = cell(k, 1);
for i = 1:k
    I = find(Args.Partition == i);
    Ic{i} = I;
    if Args.Cache
        Bc{i} = full(A(I, I)) - (K_nrm(I) * K_nrm(I)');
    else
        Ac{i} = A(I, I);
        Kc_nrm{i} = K_nrm(I);
    end
end

%% Initialize output

if isempty(U)
    [U, ~] = eigs(double(A), Args.d+1);
    U = U(:, 2:end);
    U = U ./ vecnorm(U, 2, 2);
end

clear A Am K_nrm X M

%% Initialize GPU arrays

alpha = Args.alpha;
beta = Args.beta;
if Args.GPU
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

switch Args.Solver
    case "adam"
        if Args.Verbose
            fp.head = @() fprintf("%5s %24s %12s\n", "iter", "cost val", "grad. norm");
            fp.iter = @(t, cost, grad_norm) fprintf("%5d %+.16e %12e\n", t, cost, grad_norm);
            fp.stop_cost = @() fprintf("Cost tolerance reached; tol = %g.\n", Args.Tol);
            fp.stop_grad = @() fprintf("Gradient norm tolerance reached; tol = %g.\n", Args.Tol);
            fp.stop_iter = @() fprintf("Max iter exceeded; maxiter = %g.", Args.MaxIter);
        else
            fp = struct(head = @()[], iter = @(a,b,c)[], stop_cost = @()[], stop_grad = @()[], stop_iter = @()[]);
        end
        fp.head();

        ave_grad  = zeros(size(U));   % 1st-moment estimates
        ave_grad2 = zeros(size(U));   % 2nd-moment estimates
        CostHistory = nan(1, Args.MaxIter);
        for t = 1:Args.MaxIter
            [cost, grad] = costgrad(U, Ic, Bc, Ac, Kc_nrm, M_nrm, Bm, alpha, beta);

            [U, ave_grad, ave_grad2] = adamupdate(U, grad, ave_grad, ave_grad2, t, Args.LearnRate);
            grad_norm = norm(grad, "fro");
            U = U ./ vecnorm(U,2,2);
            CostHistory(t) = cost;

            fp.iter(t, cost, grad_norm);
            if (t > 1) && (abs(cost - CostHistory(t-1)) < Args.Tol)
                fp.stop_cost(); break;
            elseif grad_norm < Args.Tol
                fp.stop_grad(); break;
            elseif t == Args.MaxIter
                fp.stop_iter(); break;
            end
        end
        CostHistory = CostHistory(~isnan(CostHistory));

    case "trustregions"
        % Create the problem structure.
        problem.M = obliquefactory(n, Args.d, "rows", Args.GPU);

        % Get the modularity matrix
        problem.costgrad = @(U) costgrad(U, Ic, Bc, Ac, Kc_nrm, M_nrm, Bm, alpha, beta);
        % checkgradient(problem);

        opts = struct(tolgradnorm=Args.Tol, maxiter=Args.MaxIter, verbosity=2*Args.Verbose);
        [U, ~, info] = trustregions(problem, U, opts);
        CostHistory = info.cost;
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

Dm = (1 - UUm);
Numm = beta * alpha * (Dm.^(2 * beta - 1));
% Denm =        alpha * ((1 - UUm).^(2 * beta));
Denm = Numm .* Dm / beta;
Cost = - sum(Bm ./ (1 + Denm), "all");

G = - 2 * Bm .* (Numm ./ (1 + Denm).^2);   % n x k
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
    Di = 1 - (Ui * Ui');
    Numi = beta * alpha * (Di.^(2 * beta - 1));
    % Deni =      alpha * (Di.^(2 * beta));
    Deni = Numi .* Di / beta;
    Cost = Cost - sum(Bi ./ (1 + Deni), "all");
    EGrad(I, :) = EGrad(I, :) - (4 * Bi .* (Numi ./ (1 + Deni).^2)) * Ui;
end

% Orthogonal projection of H in R^(nxm) to the tangent space at X.
% Compute the inner product between each column/row of H with the
% corresponding column/row of X. Remove from H the components that are
% parallel to X, by row/col.
U_dot_EGrad = sum(U .* EGrad, 2);
RGrad = EGrad - U .* U_dot_EGrad;

%% Compare to full gradient

% if 1
%     UU = U * U';
%     Num = beta * alpha * ((1 - UU).^(2 * beta - 1));
%     Den =        alpha * ((1 - UU).^(2 * beta));
%     D   =  - (8 * (A - gk * (K * K')) .* (Num ./ (1 + Den).^2)) * U;
%
%     global myvar
%     myvar = myvar + 1;
%     figure(100), axis square; hold on;
%     f = @(x, y) corr(x(:), y(:));
%     plot(myvar, f(EGrad, D), '.k')
% end

end
