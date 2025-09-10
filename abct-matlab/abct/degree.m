function S = degrees(W, type)
% DEGREES Degree of network matrix
%
%   S = degrees(W)
%   S = degrees(W, type)
%
%   Inputs:
%       W: Network matrix of size n x n.
%
%       type: Degree type
%           "first": (First) degree (default).
%           "second": Second degree.
%           "residual": Degree after global residualization.
%
%   Outputs:
%       S: Degree vector (size n).
%
%   Methodological notes:
%       The first-degree degree is the sum of connection weights. The
%       second-degree degree is the sum of squared connection weights.
%       Together, the first and second degrees are exactly or approximately
%       equivalent to several measures of network communication and control.
%
%       The residual degree is the degree after first-component removal
%       and can be approximately equivalent to the primary co-activity 
%       gradient in co-activity networks.
%
%   See also:
%       RESIDUALN, COEFVAR2.

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
        W = residualn(W, "rankone");
        S = sum(W, 2);
end
