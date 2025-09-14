function X = residualn(X, type)
% RESIDUALN Global residualization of network or data matrix
%
%   X1 = residualn(X)
%   X1 = residualn(X, type)
%
%   Inputs:
%       X:  Network matrix of size n x n, or data matrix of size n x p.
%           n is the number of nodes or data points and
%           p is the number of features.
%
%       type: Type of global residualization.
%           "degree": Degree correction (default).
%           "global": Global signal regression.
%           "rankone": Subtraction of rank-one approximation.
%
%   Outputs:
%       X1: Residual network or data matrix.
%
%   See also:
%       SHRINKAGE.

arguments
    X (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    type (1, 1) string {mustBeMember(type, ["degree", "global", "rankone"])} = "degree"
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
        
end
