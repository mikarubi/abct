function S = degrees(W, type)
% DEGREES Degree of network matrix
%
%   S = degrees(W, type)
%
%   Inputs:
%       W: Network matrix of size n x n.
%
%       type: Degree type
%           "first": (First) degree (default).
%           "second": Second degree.
%           "residual": Degree after first-mode removal.
%
%   Outputs:
%       S: Degree vector.

arguments
    W (:, :) double {mustBeNonempty, mustBeFinite, mustBeReal}
    type (1, 1) string {mustBeMember(type, ["first", "second", "residual"])} = "first"
end

switch type
    case "first"
        S = sum(W, 2);
    case "second"
        S = sum(W.^2, 2);
    case "residual"
        W = moderemoval(W, "mode");
        S = sum(W, 2);
end
