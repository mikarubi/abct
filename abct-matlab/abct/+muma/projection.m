function U2 = projection(U3)
% Mercator projection

[X, Y, Z] = deal(U3(:, 1), U3(:, 2), U3(:, 3));
U2 = [atan2(Y, X), log((1 + Z)./(1 - Z))/2];

end
