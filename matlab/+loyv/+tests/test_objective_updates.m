function test_objective_updates(Args, W, M, My, Vx, Vy, U, MU, delta_QU)
% Test accuracy of objective update rules for Nodes U in Modules MU

% Unpack arguments
k = Args.k;
n = length(M);
M0 = M;

delta_QU1 = zeros(k, numel(U));
for i = 1:numel(U)
    u = U(i);
    mu = MU(i);
    cii_nrm = zeros(k, 1);

    M = M0;
    for mv = 1:k
        M(u) = mv;

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
                    Sm = sum(Smn, 2);                   % degree of module
                end

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
            case "kmeans"
                Cii_nrm = Cii ./ N;
            case "spectral"
                Cii_nrm = Cii ./ Sm;
            case "cokmeans"
                Cii_nrm = Cii ./ sqrt(N .* Ny);
            case "cospectral"
                Cii_nrm = Cii ./ sqrt(Dii .* Eii);
        end

        cii_nrm(mv) = sum(Cii_nrm);
    end
    delta_QU1(:, i) = cii_nrm - cii_nrm(mu);
end

assert(max(abs(delta_QU1 - delta_QU), [], "all") < Args.tolerance)
