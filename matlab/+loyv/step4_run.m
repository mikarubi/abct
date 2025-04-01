function [M, Q] = step4_run(Args, W, M, My, Vx, Vy)
% Loyvain main algorithm

% Unpack arguments
k = Args.k;

n = length(M);
LinIdx = M + k*(0:n-1);                         % two-dimensional indices of M

MM = sparse(M, 1:n, 1, k, n);                   % two-dimensional representation
N = full(sum(MM, 2));                           % number of nodes in module
if Args.similarity == "network"
    Smn = MM * W;                               % degree of module to node
elseif Args.method == "coloyvain"
    ny = length(My);
    MMy = sparse(My, 1:ny, 1, k, ny);           % two-dimensional representation
    Ny = full(sum(MMy, 2));
    Smn = MMy * W';
else
    X = Args.X;
    G = MM * X;                                 % cluster centroid
    Smn = G * X';                               % dot of centroid with node
end
if Args.objective == "spectral"
    Sn = sum(Smn, 1);                           % degree of node
    Sm = sum(Smn, 2);                           % degree of module
end

switch Args.method
    case "loyvain"
        Wii = Args.Wii;                         % within-node weight sum
        Cii = diag(Smn * MM');                  % within-module weight sum
    case "coloyvain"
        Vii = diag(Vx);
        Cii = diag(MM * Smn');                  % within-module weight sum
        if Args.objective == "cospectral"
            Dii = diag(MM  * Vx * MM');         % within-module weight sum of X
            Eii = diag(MMy * Vy * MMy);         % within-module weight sum of Y
        end
end

switch Args.objective
    case "kmeans"
        Cii_nrm = Cii ./ N;
    case "spectral"
        Cii_nrm = Cii ./ Sm;
    case "cokmeans"
        Cii_nrm = Cii ./ sqrt(N .* Ny);
    case "cospectral"
        Cii_nrm = Cii ./ sqrt(Dii .* Eii);
end

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
            case "cokmeans"
                delta_QU = ...
                    (Cii      + Smn(:, U)) ./ sqrt((N      + 1) .* Ny     ) - Cii_nrm + ...
                    (Cii(MU)' - Smn(LinU)) ./ sqrt((N(MU)' - 1) .* Ny(MU)') - Cii_nrm(MU)';
            case "cospectral"
                delta_QU = ...
                    (Cii      + Smn(:, U)) ./ sqrt((Dii       + 2 * Smn(:, U) - Vii(U)) .* Eii) - Cii_nrm + ...
                    (Cii(MU)' - Smn(LinU)) ./ sqrt((Dii(MU)' -  2 * Smn(LinU) - Vii(U)) .* Eii) - Cii_nrm(MU)';
        end
        delta_QU(:, N(MU) == 1) = - inf;        % no change allowed if one-node cluster
        delta_QU(MU + k*(0:b-1)) = 0;           % no change if node stays in own module

        % Update if improvements
        [max_delta_QU, MU_new] = max(delta_QU);
        if max(max_delta_QU) > Args.tolerance
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

            % Get Cii, C_nrm, and Dm
            switch Args.method
                case "loyvain"
                    Cii = diag(Smn * MM');              % within-module weight sum
                    % Update G and Dmn
                    if Args.similarity == "network"
                        delta_Dmn = delta_MMI * W(I, :);
                    else
                        delta_G = delta_MMI * X(I, :);  % change in centroid
                        G = G + delta_G;                % update centroids
                        delta_Dmn = delta_G * X';       % change in degree of module to node
                    end
                    Smn = Smn + delta_Dmn;              % update degree of module to node
                    if Args.objective == "spectral"
                        Sm = Sm + sum(delta_Dmn, 2);
                    end
                case "coloyvain"
                    Cii = diag(MM * Smn');              % within-module weight sum
                    if Args.objective == "cospectral"
                        Dii = diag(MM * Vx * MM');      % within-module weight sum of X
                    end
            end

            switch Args.objective
                case "kmeans"
                    Cii_nrm = Cii ./ N;
                case "spectral"
                    Cii_nrm = Cii ./ Sm;
                case "cokmeans"
                    Cii_nrm = Cii ./ sqrt(N .* Ny);
                case "cospectral"
                    Cii_nrm = Cii ./ sqrt(Dii .* Eii);
            end
        end
    end
    if max_delta_Q < Args.tolerance
        break
    end
    if Args.display == "iteration"
        fprintf("Replicate: %4d.    Iteration: %4d.    Largest \x0394: %4.4f\n", ...
            Args.replicate_i, v, max_delta_Q)
    end
    if v == Args.maxiter
        warning("Algorithm did not converge after %d iterations.", v)
    end
end

% Return objective
Q = sum(Cii_nrm);

end
