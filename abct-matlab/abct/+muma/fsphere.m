function V = fsphere(p)
% Unit Fibonacci sphere
I = (0:p-1).' + 0.5;
golden_ratio = (1 + sqrt(5))/2;
Phi = acos(1 - 2*I/p);
Theta = 2*pi*I / golden_ratio;
V = [cos(Theta).*sin(Phi), sin(Theta).*sin(Phi), cos(Phi)];

end
