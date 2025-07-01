function [M1, Q1] = louvains(A, gamma, M0)

n = length(A);                                                  % number of nodes
if ~exist('gamma','var') || isempty(gamma)
    gamma = 1;
end
if ~exist('M0','var') || isempty(M0)
    M0 = (1:n)';
end
% assert(islogical(A))
assert(issparse(A))
assert(issymmetric(A))
assert(~any(diag(A)))

s = sum(A, "all");
gamma_s = gamma / s;
tol = 1e-6;
A = double(A);

[~, ~, M] = unique(M0);                                         % initial community structure
M1 = M;                                                         % global community structure

Q0 = -inf;
first_pass = true;
while 1
    n = length(A);
    J = arrayfun(@(i) find(A(:, i)), 1:n, uniformoutput=false); % neighbors

    M = (1:n)';                                                 % initial modules
    K_nrm = sqrt(gamma_s) * full(sum(A, 2));                    % degree
    Km_nrm = K_nrm;                                             % module degree
    Bd = full(diag(A)) - (K_nrm.^2);                            % self modularities
    Q1 = sum(Bd) / s;                                           % total modularity
    if Q1 - Q0 <= tol
        break;
    else
        Q0 = Q1;
    end

    all_neg_bd = all(Bd < 0);
    flag = true;                                                % flag for within-hierarchy search
    while flag
        n = length(A);
        Queued = randperm(n);                                   % queued nodes
        while any(Queued)
            [~, P] = sort(Queued);                              % get node queue order
            P = P(logical(Queued(P)));                          % keep nodes in queue
            p = length(P);
            Queued(P) = 1:p;                                    % reset queue numbering

            flag = false;
            for h = 1:p                                         % loop over all nodes in random order
                i = P(h);
                Queued(i) = 0;

                u = M(i);                                       % module of node
                Ji = J{i};
                if all_neg_bd                                   % get module of neighbors
                    V0 = [M(Ji); u];                            % redundant self-modules Ok because Bd < 0
                else
                    V0 = [setdiff(M(Ji), u); u];                % setdiff self-modules because Bd can be > 0
                end
                Ami = accumarray(M(Ji), A(Ji, i), [max(V0), 1]);
                dQ = full(Ami(V0)) - K_nrm(i) * Km_nrm(V0);
                dQ = dQ - dQ(end) + Bd(i);
                dQ(end) = 0;                                    % set self-module to 0 as escape

                [max_dQ, idx] = max(dQ);                        % maximal increase in modularity and corresponding module
                v = V0(idx);                                    % convert to actual module representation
                if max_dQ > tol                                 % if maximal increase is positive
                    M(i) = v;                                   % reassign module
                    Km_nrm(v) = Km_nrm(v) + K_nrm(i);
                    Km_nrm(u) = Km_nrm(u) - K_nrm(i);

                    flag = true;                                % If we move the node to a different community, we add
                    Pi = Ji(M(Ji) ~= v)';                       % to the rear of the queue all neighbours of the node
                    Pi = Pi(~Queued(Pi));                       % that do not belong to the node’s new community and
                    Queued(Pi) = p + (1:length(Pi));            % that are not yet in the queue (Traag et al., 2019)
                    p = p + length(Pi);
                end
            end
        end
    end
    [~, ~, M] = unique(M);                                      % new module assignments

    M0 = M1;
    if first_pass
        M1 = M;
        first_pass = false;
    else
        for i = 1:n                                             % loop through initial module assignments
            M1(M0==i) = M(i);                                   % assign new modules
        end
    end

    L = sparse(1:n, M, 1);                                      % new module assignments
    A = full(L' * A * L);                                       % node-to-module strength
end

end
