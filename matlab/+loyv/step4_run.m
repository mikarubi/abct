function [M, Q] = step4_run(Args, W, M, My, Vx, Vy, Vii)
% Loyvain main algorithm

% Unpack arguments
k = Args.k;

n = length(M);
LinIdx = M + k*(0:n-1);                         % two-dimensional indices of M

MM = sparse(M, 1:n, 1, k, n);                   % two-dimensional representation
N = full(sum(MM, 2));                           % number of nodes in module
switch Args.method
    case "loyvain"
        if Args.similarity == "network"
            Smn = MM * W;                       % degree of module to node
        else
            X = Args.X;
            G = MM * X;                         % cluster centroid
            Smn = G * X';                       % dot of centroid with node
        end
        Cii = diag(Smn * MM');                  % within-module weight sum
        if Args.objective == "spectral"
            Sn = sum(Smn, 1);                   % degree of node
            Sm = sum(Smn, 2);                   % degree of module
        end
        Wii = Args.Wii;                         % within-node weight sum

    case "coloyvain"
        ny = length(My);
        MMy = sparse(My, 1:ny, 1, k, ny);       % two-dimensional representation
        Ny = full(sum(MMy, 2));
        Smn = MMy * W';                         % strength node to module of Wxy
        Cii = diag(MM * Smn');                  % within-module weight sum
        if Args.objective == "cospectral"
            Tmn = MM * Vx;                      % strength node to module of Wxx
            Dii = diag(Tmn * MM');              % within-module weight sum of X
            Eii = diag(MMy * Vy * MMy');        % within-module weight sum of Y
        end
end

switch Args.objective
    case "kmeans";      Cii_nrm = Cii ./ N;
    case "spectral";    Cii_nrm = Cii ./ Sm;
    case "cokmeans";    Cii_nrm = Cii ./ sqrt(N .* Ny);
    case "cospectral";  Cii_nrm = Cii ./ sqrt(Dii .* Eii);
end

if (k == 1) || (k == n)
    Args.maxiter = 0;                           % skip loop if trivial partition
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
                    (Cii      + Smn(:, U)) ./ sqrt((N     + 1)  .* Ny     ) - Cii_nrm + ...
                    (Cii(MU)' - Smn(LinU)) ./ sqrt((N(MU) - 1)' .* Ny(MU)') - Cii_nrm(MU)';
            case "cospectral"
                delta_QU = ...
                    (Cii      + Smn(:, U)) ./ sqrt((Dii      + 2 * Tmn(:, U) + Vii(U)) .* Eii)      - Cii_nrm + ...
                    (Cii(MU)' - Smn(LinU)) ./ sqrt((Dii(MU)' - 2 * Tmn(LinU) + Vii(U)) .* Eii(MU)') - Cii_nrm(MU)';
        end
        delta_QU(:, N(MU) == 1) = - inf;        % no change allowed if one-node cluster
        delta_QU(MU + k*(0:b-1)) = 0;           % no change if node stays in own module

        % UNCOMMENT TO TEST OBJECTIVE UPDATES with runtests loyv.tests.test_options
        for name = ["My" "Vx" "Vy"]
            if ~exist(name, "var"); eval(name + " = [];"); end
        end
        loyv.tests.test_objective_updates(Args, W, M, My, Vx, Vy, U, MU, delta_QU)
        % END UNCOMMENT TO TEST OBJECTIVE UPDATES

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

            % Update all relevant variables
            N = N_new;
            M(I) = MI_new;
            MM = sparse(M, 1:n, 1, k, n);
            LinIdx(I) = MI_new + k*(I-1);
            switch Args.method
                case "loyvain"
                    % Update G and Smn
                    if Args.similarity == "network"
                        delta_Smn = delta_MMI * W(I, :);
                    else
                        delta_G = delta_MMI * X(I, :);  % change in centroid
                        G = G + delta_G;                % update centroids
                        delta_Smn = delta_G * X';       % change in degree of module to node
                    end
                    Smn = Smn + delta_Smn;              % update degree of module to node
                    if Args.objective == "spectral"
                        Sm = Sm + sum(delta_Smn, 2);
                    end
                    Cii = diag(Smn * MM');              % within-module weight sum
                case "coloyvain"
                    Cii = diag(MM * Smn');              % within-module weight sum
                    if Args.objective == "cospectral"
                        delta_Tmn = delta_MMI * Vx(I, :);
                        Tmn = Tmn + delta_Tmn;
                        Dii = diag(Tmn * MM');          % within-module weight sum of X
                    end
            end
            switch Args.objective
                case "kmeans";      Cii_nrm = Cii ./ N;
                case "spectral";    Cii_nrm = Cii ./ Sm;
                case "cokmeans";    Cii_nrm = Cii ./ sqrt(N .* Ny);
                case "cospectral";  Cii_nrm = Cii ./ sqrt(Dii .* Eii);
            end
            % UNCOMMENT TO TEST VARIABLE UPDATES with runtests loyv.tests.test_options
            Vals = struct();
            for name = ["N" "M" "MM" "LinIdx" "Smn" "Sm" "Tmn" "Dii" "Cii" "Cii_nrm"]
                if ~exist(name, "var"); val = []; else; eval("val = " + name + ";"); end
                Vals.(name) = val;
            end
            loyv.tests.test_variable_updates(Args, W, M, My, Vx, Vy, Vals)
            % END UNCOMMENT TO TEST VARIABLE UPDATES
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
