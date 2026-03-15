function X = residualn(X, type)
% RESIDUALN Residualization of network or data matrix
%
%   W1 = residualn(W)
%   W1 = residualn(W, type)
%   X1 = residualn(X)
%   X1 = residualn(X, type)
%
%   Inputs:
%       W:  Network matrix of size n x n.
%       OR
%       X:  Data matrix of size n x p, where
%           n is the number of data points and
%           p is the number of features.
%
%       type: Type of residualization.
%           "degree": Degree correction (default)
%               Subtraction of the rescaled product of the degrees.
%           "degree_ctr": Double centering
%               Subtraction of the rescaled and shifted degrees.
%           "global": Global signal regression
%               Regression out of the global signal (column mean).
%           "global_ctr": Global signal subtraction (centering)
%               Subtraction of the global signal (column mean).
%           "rankone": Rank-one subtraction
%               Subtraction of the rank-one approximation.
%
%   Outputs:
%       W1: Residual network matrix.
%       OR
%       X1: Residual data matrix.
%
%   See also:
%       SHRINKAGE.

arguments
    X (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    type (1, 1) string {mustBeMember(type, ...
        ["degree", "degree_ctr", "global", "global_ctr", "rankone"])} = "degree"
end

if ismember(type, ["degree", "degree_ctr"])
    So = mean(X, 2);        % NB: mean not sum
    Si = mean(X, 1);        % NB: mean not sum
    s = mean(X, "all");     % NB: mean not sum
    switch type
        case "degree"       % Degree correction
            assert(all(X >= 0, "all"), ...
                "Invalid degree correction: Matrix must be non-negative.")
            X = X - So * Si / s;
        case "degree_ctr"   % Double centering
            X = X - So - Si + s;
    end

elseif ismember(type, ["global", "global_ctr"])
    G = mean(X, 1);
    switch type
        case "global"       % Global signal regression
            X = X - (X * G') * G / (G * G');
            % (X - (X * G') * G / (G * G')) * G' = 0    % verification
        case "global_ctr"   % Global signal subtraction
            X = X - G;
    end

elseif type == "rankone"    % Subtraction of rank-one approximation
    [U, S, V] = svds(X, 1);
    X = X - U * S * V';

end
