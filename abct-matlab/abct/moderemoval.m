function X = moderemoval(X, type)
% MODEREMOVAL Mode removal from network or timeseries data
%
%   X1 = moderemoval(X)
%   X1 = moderemoval(X, type)
%
%   Inputs:
%       X:  Network matrix of size n x n, or data matrix of size n x p.
%           n is the number of nodes or data points and
%           p is the number of features.
%
%       type: Type of mode removal.
%           "degree": Degree correction (default).
%           "global": Global signal regression.
%           "rankone": Subtraction of rank-one approximation.
%           "soft": Soft removal of primary modes.
%
%   Outputs:
%       X1: Network or timeseries matrix after mode removal.
%
%   Methodological notes:
%       Degree correction, global signal regression, and subtraction of
%       rank-one approximation all produce approximatley equivalent
%       results. The "soft" mode removal makes the network sparse by using
%       cubic interpolation to "despike" an initial eigenspectrum peak.
%
%   See also:
%       DEGREES, GRADIENTS.

arguments
    X (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    type (1, 1) string {mustBeMember(type, ["degree", "global", "rankone", "soft"])} = "degree"
end

switch type
    % Degree correction
    case "degree"
        assert(all(X >= 0, "all"), ...
            "Invalid degree correction: Matrix must be non-negative.")
        So = sum(X, 2);
        Si = sum(X, 1);
        X = X - So * Si / sum(So);

    % Global signal regression
    case "global"
        G = mean(X, 1);
        X = X - (X * G') * G / (G * G');
        % (X - (X * G') * G / (G * G')) * G' = 0    % verification

    % Subtraction of rank-one approximation
    case "rankone"
        [U, S, V] = svds(X, 1);
        X = X - U * S * V';

    % Soft removal of primary modes
    case "soft"
        assert(isequal(size(X, 1), size(X, 2)) && all(X - X' < eps("single"), "all"), ...
            "Invalid soft mode removal: Network matrix must be symmetric.")
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
        X = V * diag(D) * V';
end
