function test_variable_updates(Args, W, M, My, Vx, Vy, Vals)
%#ok<*NASGU> Test accuracy of variable update rules after an update

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

for name = reshape(string(fieldnames(Vals)), 1, [])
    if ~isempty(Vals.(name))
        eval("true_val = " + name + ";")
        assert(max(abs(true_val - Vals.(name)), [], "all") < Args.tolerance)
    end
end
