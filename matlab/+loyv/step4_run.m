function [M, Q] = step4_run(Args, M, replicate_i)
% Loyvain main algorithm

% Unpack arguments
X = Args.X;
W = Args.W;
k = Args.k;
n = Args.n;
Wii = Args.Wii;

LinIdx = M + k*(0:n-1);                         % two-dimensional indices of M

MM = sparse(M, 1:n, 1, k, n);                   % two-dimensional representation
N = full(sum(MM, 2));                           % number of nodes in module
if Args.similarity == "network"
    Smn = MM * W;                               % degree of module to node
else
    G = MM * X;                                 % cluster centroid
    Smn = G * X';                               % dot of centroid with node
end
Cii = diag(Smn * MM');                          % within-module weight sum

switch Args.objective
    case "kmeans"
        Cii_nrm = Cii ./ N;
    case "spectral"
        Sn = sum(Smn, 1);                       % degree of node
        Sm = sum(Smn, 2);                       % degree of module
        Cii_nrm = Cii ./ Sm;
end

tol = 1e-10;
for v = 1:Args.maxiter

    max_delta_Q = 0;                            % maximal increase over all batches
    Batches = mat2cell(randperm(n), 1, diff(round(0:n/Args.numbatches:n)));
    for u = 1:Args.numbatches
        U = Batches{u};                         % indices of nodes in batch
        LinU = LinIdx(U);                       % linear indices of nodes in batch
        MU = M(U);                              % module assignments of nodes in batch
        b = numel(U);                           % number of nodes in batch

        switch Args.objective
            case "kmeans"
                delta_QU = ...
                    ((2 * Smn(:, U) + Wii(U)) - Cii_nrm     ) ./ (N      + 1) - ...
                    ((2 * Smn(LinU) - Wii(U)) - Cii_nrm(MU)') ./ (N(MU)' - 1);
            case "spectral"
                delta_QU = ...
                    ((2 * Smn(:, U) + Wii(U)) - Cii_nrm      .* Sn(U)) ./ (Sm      + Sn(U)) - ...
                    ((2 * Smn(LinU) - Wii(U)) - Cii_nrm(MU)' .* Sn(U)) ./ (Sm(MU)' - Sn(U));
        end
        delta_QU(:, N(MU) == 1) = - inf;        % no change allowed if one-node cluster
        delta_QU(MU + k*(0:b-1)) = 0;           % no change if node stays in own module

        % Update if improvements
        [max_delta_QU, MU_new] = max(delta_QU);
        if max(max_delta_QU) > tol
            max_delta_Q = max(max_delta_Q, max(max_delta_QU));

            IU = find(MU ~= MU_new);            % batch indices of nodes to be switched
            I = U(IU);                          % actual indices of nodes to be switched
            MI_new = MU_new(IU);                % new module assignments

            % get delta modules and ensure non-empty modules
            n_i = numel(I);
            MMI = sparse(M(I), 1:n_i, 1, k, n_i);
            while 1
                MMI_new = sparse(MI_new, 1:n_i, 1, k, n_i);
                delta_MMI = (MMI_new - MMI);
                N_new = N + sum(delta_MMI, 2);
                if all(N_new)
                    break;
                else
                    E = find(~N_new);           % empty modules
                    k_e = numel(E);             % number of empty modules
                    MI_new(randperm(n_i, k_e)) = E;
                end
            end

            % Update N, M, MM, and LinIdx
            N = N_new;
            M(I) = MI_new;
            MM = sparse(M, 1:n, 1, k, n);
            LinIdx(I) = MI_new + k*(I-1);

            % Update G and Dmn
            if Args.similarity == "network"
                delta_Dmn = delta_MMI * W(I, :);
            else
                delta_G = delta_MMI * X(I, :);  % change in centroid
                G = G + delta_G;                % update centroids
                delta_Dmn = delta_G * X';       % change in degree of module to node
            end
            Smn = Smn + delta_Dmn;              % update degree of module to node

            % Get Cii, C_nrm, and Dm
            Cii = diag(Smn * MM');              % within-module weight sum
            switch Args.objective
                case "kmeans"
                    Cii_nrm = Cii ./ N;
                case "spectral"
                    Sm = Sm + sum(delta_Dmn, 2);
                    Cii_nrm = Cii ./ Sm;
            end
        end
    end
    if max_delta_Q < tol
        break
    end
    if Args.display == "iteration"
        fprintf("Replicate: %4d.    Iteration: %4d.    Largest \x0394: %4.4f\n", ...
            replicate_i, v, max_delta_Q)
    end
    if v == Args.maxiter
        warning("Algorithm did not converge after %d iterations.", v)
    end
end

% Return objective
Q = sum(Cii_nrm);

end
