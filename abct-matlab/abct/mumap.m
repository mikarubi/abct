function [U, CostHistory, Args] = mumap(X, U, Args)
% MUMAP

arguments
    X (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    U (:, :) = []
    Args.solver (1, 1) ...
        string {mustBeMember(Args.solver, ["adam", "trustregions"])} = "adam";
    Args.k (1, 1) double {mustBeNonempty, mustBeFinite, mustBeReal} = 10
    Args.alpha (1, 1) double {mustBePositive} = 2
    Args.beta (1, 1) double {mustBePositive} = 1/2
    Args.gamma (1, 1) double {mustBePositive} = 1
    Args.Partition (:, 1) double {mustBeInteger, mustBePositive} = [];
    Args.MaxIter (1, 1) double {mustBeInteger,mustBePositive} = 1e4
    Args.LearnRate (1, 1) double {mustBePositive} = 1e-3
    Args.Tol (1, 1) double {mustBePositive} = 1e-6
    Args.Verbose (1, 1) logical = true;
    Args.GPU (1,1) logical = false
end

if Args.GPU
    assert(license("test", "distrib_computing_toolbox") && gpuDeviceCount, ...
        "GPU use requires Parallel Computing Toolbox and active GPU.")
end
switch Args.solver
    case "adam"
        assert(license("test", "neural_network_toolbox"), ...
            "Adam solver requires Deep Learning Toolbox.")
    case "trustregions"
        assert((exist("obliquefactory", "file") && exist("trustregions", "file")), ...
            "Manopt functions not found. Call importmanopt to setup.")
end

m = 3;
n = size(X, 1);

if issparse(X)
    A = X;
else
    % Generate a common neighbors matrix
    Col = knnsearch(X, X, K=Args.k+1, Distance="correlation");
    Col = reshape(Col, [], 1);
    Row = (1:n).';
    Row = reshape(Row(:, ones(1,Args.k+1)), [], 1);
    A = sparse([Row; Col], [Col; Row], true);
end
A(1:n+1:end) = 0;

% Get module structure and partition matrix
[Args.Partition, Args.Q] = louvains(A, Args.gamma);
M = sparse(1:n, Args.Partition, 1);

% Get normalized degrees vector
K = sum(A, 2);

if isempty(U)
    [U, ~] = eigs(double(A), m+1);
    U = U(:, 2:end);
    U = U ./ vecnorm(U, 2, 2);
end

alpha = Args.alpha;
beta = Args.beta;
gamma = Args.gamma;
if Args.GPU
    A = gpuArray(A);
    K = gpuArray(K);
    U = gpuArray(U);
    M = gpuArray(M);
    alpha = gpuArray(Args.alpha);
    beta = gpuArray(Args.beta);
    gamma = gpuArray(Args.gamma);
end

switch Args.solver
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

        ave_grad  = zeros(size(U));   % 1st‑moment estimates
        ave_grad2 = zeros(size(U));   % 2nd‑moment estimates
        CostHistory = nan(1, Args.MaxIter);
        for t = 1:Args.MaxIter
            [cost, grad] = costgrad(U, A, K, M, alpha, beta, gamma);

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
        problem.M = obliquefactory(n, m, "rows", Args.GPU);

        % Get the modularity matrix
        problem.costgrad = @(U) costgrad(U, A, K, M, alpha, beta, gamma);

        % checkgradient(problem);
        opts = struct(tolgradnorm=Args.Tol, maxiter=Args.MaxIter, verbosity=2*Args.Verbose);
        [U, ~, info] = trustregions(problem, U, opts);
        CostHistory = info.cost;
end

end

function [Cost, RGrad] = costgrad(U, A, K, M, alpha, beta, gamma)

%% Compute mean-field between-module cost and gradient

% Bm == (A - g * ones(n)) * M
g = gamma * mean(A, "all");     % mean(K)^2 / sum(K);
N = sum(M);

Am = A * M;
% Bm == ((A - g) .* (~(M * M'))) * M
%    == ((A - g) * M - ((A - g) .* (M * M')) * M
%    -> then simplify (A - g) .* (M * M')
Bm = ((Am - M .* Am) - full(g * N - g .* (M .* N)));

% UUm == ((U * U') .* (~(M * M'))) * Mn
Mn = (M./N);
UUm = U * (U' * Mn);
UUm = UUm - M .* UUm;           % exclude self-modules, as above.

Numm = beta * alpha * ((1 - UUm).^(2 * beta - 1));
Denm =        alpha * ((1 - UUm).^(2 * beta));
Cost = - sum(Bm .* ((1 - Denm) ./ (1 + Denm)), "all");

G    = -4 * Bm .* (Numm ./ (1 + Denm).^2);   % n × k
EGrad  = G * (Mn' * U) + Mn * (G' * U);

%% Compute full within-module cost and gradient

k = size(M, 2);
gk = gamma / sum(K);
for i = 1:k
    I = logical(M(:, i));
    Ki = K(I);
    Bi = A(I, I) - gk * (Ki * Ki');

    Ui = U(I, :);
    UUi = Ui * Ui';
    Numi = beta * alpha * ((1 - UUi).^(2 * beta - 1));
    Deni =        alpha * ((1 - UUi).^(2 * beta));
    Cost = Cost - sum(Bi .* ((1 - Deni) ./ (1 + Deni)), "all");
    EGrad(I, :) = EGrad(I, :) - (8 * Bi .* (Numi ./ (1 + Deni).^2)) * Ui;
end

% Orthogonal projection of H in R^(nxm) to the tangent space at X.
% Compute the inner product between each column/row of H with the
% corresponding column/row of X. Remove from H the components that are
% parallel to X, by row/col.
U_dot_EGrad = sum(U.*EGrad, 2);
RGrad = EGrad - U.*U_dot_EGrad;

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

