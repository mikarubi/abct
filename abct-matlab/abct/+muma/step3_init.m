function U = step3_init(Args)
% m-umap output initialization

%% Initialize output

A = Args.A;
M = Args.M;
Am = Args.Am;
k = Args.k;
switch Args.start
    case "custom"
        U = Args.U;
        U = U ./ vecnorm(U, 2, 2);
    case "spectral_nn"                          % spectral on knn matrix
        [U, ~] = eigs(double(A), Args.d+1);
        U = U(:, 2:end);
        U = U ./ vecnorm(U, 2, 2);
    case "spectral"                             % spectral on modules
        Amm = full(M' * Am);
        Kmm = sum(Amm, 2);
        Bmm = Amm - Kmm * Kmm' / sum(Amm, "all");
        [Um, ~] = eigs(Bmm, Args.d);
        Um = Um ./ vecnorm(Um, 2, 2);
        U = Um(Args.partition, :);
    case "greedy"                               % spherical maximin
        Amm = full(M' * Am);                    % module connectivity
        Amm(1:k+1:end) = nan;                   % ignore self-connections
        Kmm_ = zeros(k, 1);                     % degree to placed modules
        Um = zeros(k, Args.d);                  % module locations
        Vm = muma.fsphere(k);                   % Fibonacci sphere
        [~, ux] = max(sum(Amm, 2, "omitnan"));  % initial module index
        [~, vx] = max(sum(Vm * Vm', 2));        % initial location index
        for i = 1:k
            Um(ux, :) = Vm(vx, :);              % assign location
            Vm(vx, :) = nan;                    % remove point from consideration
            Kmm_ = Kmm_ + Amm(:, ux);           % add module connectivity (with self-nan's)
            [~, ux] = min(Kmm_);                % least connected module (nan's in Kmm mask set modules)
            [~, vx] = min(Vm * mean(Um)');      % furthest location (nan's in Vm mask used locations)
        end
        U = Um(Args.partition, :);
end

end
