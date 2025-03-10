function X = moderemoval(X, method)
% MODEREMOVAL Mode removal from network or timeseries
%
%   X1 = moderemoval(X, "degree")
%   X1 = moderemoval(X, "global")
%   X1 = moderemoval(X, "mode")
%   X1 = moderemoval(X, "soft")
%
%   Inputs:
%       X: Input network or timeseries matrix.
%
%       method: Method for mode removal.
%           "degree": Degree correction (default).
%           "global": Global signal regression.
%           "mode": Subtraction of rank-one approximation.
%           "soft": Soft removal of primary modes.
%
%   Outputs:
%       X1: Network or timeseries matrix with mode removed.

arguments
    X (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    method (1, 1) string {mustBeMember(method, ["degree", "global", "mode", "soft"])} = "degree"
end

switch method
    case "degree"
        %% Degree correction
        assert(all(X >= 0, "all"), "Input matrix must be non-negative.")
        So = sum(X, 2);
        Si = sum(X, 1);
        X = X - So * Si / sum(So);

    case "global"
        %% Global signal regression
        G = mean(X, 1);
        X = X - (X * G') * G / (G * G');
        % (X - (X * G') * G / (G * G')) * G' = 0    % verification

    case "mode"
        %% Subtraction of rank-one approximation
        [U, S, V] = svds(X, 1);
        X = X - U * S * V';
        
    case "soft"
        %% Soft removal of primary modes
        assert(issymmetric(X), "Input matrix must be symmetric.")
        [V, D] = eig(X, "vector");
        [~, ix] = sort(D, "descend");
        V = V(:, ix);
        D = D(ix);

        n = length(D);
        [bk, r0] = polyfit((1:n)', D, 3);
        rms0 = r0.normr / sqrt(n);    % get rms
        x = rescale(1:n).';
        y(n) = 0;
        for k = 1:n
            b = bk;
            [bk, rk] = polyfit((k:n).', D(k:n), 3);
            assert(numel((k:n))==(n-k+1))
            rmsk = rk.normr / sqrt(n-k+1);
            y(k) = (rms0 - rmsk) / rms0;
            % detect knee of optimal fit and break
            if k > 1 && ((y(k) - x(k)) < (y(k-1) - x(k-1)))
                break
            end
        end

        D = polyval(b, (1:n).');
        X = V * D * V';
end
