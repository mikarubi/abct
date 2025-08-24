function [U3, R] = rotation(U3)

% ensure spherical geometry
n = size(U3, 1);
U3 = U3 ./ vecnorm(U3, 2, 2);

%% Rotate data to empty poles

p = 144;                    % Fibonacci number
V = muma.fsphere(p);        % Unit Fibonacci sphere

% Poles have minimal (maximal correlations to all other points)
D = max(abs(U3 * V'), [], 1);
[~, idx] = min(D);

% Rotate data so that v becomes [0, 0, 1]
v = V(idx, :);
Rp = rota(v, 3);
U3 = U3 * Rp;

%% Rotate data to sparse anti-prime meridian

% Unit circle
Theta = pi * (-p:2:p-2).'/p;
V = [cos(Theta) sin(Theta)];

% Sparse meridians have few strong correlations to other points
alpha = 12*pi/p;         % degree band (6/p in either direction)
D = zeros(p, 1);
for i = 1:p
    Vi = [V(i*ones(n, 1), :), U3(:, 3)];
    Vi = Vi ./ vecnorm(Vi, 2, 2);
    % Number of points within the angle boundary
    D(i) = mean(sum(U3 .* Vi, 2) > cos(alpha));
end

% Find middle meridian among several identical
Comps  = diff([0; [D; D]==min(D); 0]);                % Enforce circular boundary conditions
Comp_sta = find(Comps ==  1);
Comp_fin = find(Comps == -1);
[~, u] = max(Comp_fin - Comp_sta);
idx = floor((Comp_sta(u) + Comp_fin(u))/2);
idx = mod(idx-1, p) + 1;
v = [V(idx, :) 0];

% Rotate U3 so that v is placed on [-1, 0, 0]
Rm = rota(-v, 1);
U3 = U3 * Rm;

% Combine rotations
R = Rp * Rm;

end

function R = rota(v, a)
% Rotate data such that v aligns with a basis vector
epss = double(eps("single"));

e = zeros(1, 3);
e(a) = 1;
b = find(~e, 1);
if norm(abs(e) - abs(v)) < epss        % Target vector parallel
    R = eye(3);
    if norm(e - (- v)) < epss          % Target vector antiparallel
        R(a, a) = -1;
        R(b, b) = -1;
    end
else
    % Cross product
    x = cross(e, v);
    cos_theta = dot(e, v);
    sin_theta = norm(x);

    % Skew-symmetric matrix
    Q = [0    -x(3)  x(2);
        x(3)    0   -x(1);
        -x(2)  x(1)   0 ];

    % Rodrigues formula
    R = eye(3) + Q + Q * Q * ((1 - cos_theta)/(sin_theta^2));
end

% Check that rotation matrix is valid and rotation is correct
assert((norm(eye(3) - R' * R) < epss) && (abs(1 - det(R)) < epss))
assert(norm(e - v * R) < epss)

end
