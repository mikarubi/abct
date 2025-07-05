function [M, Q] = louvains(A, Args)
arguments
    A (:, :) double {mustBeNonnegativeSymmetric}
    Args.gamma (1, 1) {mustBePositive} = 1
    Args.start (:, 1) {mustBePositive, mustBeInteger} = (1:length(A)).'
    Args.replicates (1, 1) {mustBeInteger, mustBePositive} = 10
    Args.tolerance (1, 1) double {mustBePositive} = 1e-10
    Args.finaltune (1, 1) logical = false
    Args.display (1, 1) string {mustBeMember(Args.display, ["none", "replicate"])} = "none"
end
if any(diag(A))
    warning("Input matrix has a non-empty diagonal. Louvains may be slow.")
end

Q = -inf;
for i = 1:Args.replicates
    Args.replicate_i = i;

    % run algorithm
    [M1, Q1] = louvains_run(A, Args);
    if Args.finaltune
        Args1 = Args;
        Args1.start = M1;
        [M1, Q1] = louvains_run(A, Args1);
    end

    % test for increase
    if (Q1 - Q) > Args.tolerance
        if ismember(Args.display, ["replicate", "iteration"])
            fprintf("Replicate: %4d.    Objective: %4.4f.    \x0394: %4.4f.\n", i, Q1, Q1 - Q);
        end
        Q = Q1;
        M = M1;
    end
end

end
function mustBeNonnegativeSymmetric(A)
assert(issymmetric(A) && all(nonzeros(A) >= 0), ...
    "Matrix must be nonnegative and symmetric.");
end

function [M1, Q1] = louvains_run(A, Args)

n = length(A);                                                  % number of nodes
s = sum(A, "all");                                              % sum of all edges
gamma_s = Args.gamma / s;                                       % normalization constant

[~, ~, M] = unique(Args.start);                                 % initial community structure
M1 = M;                                                         % global community structure

Q0 = -inf;
first_pass = true;
while 1
    J = arrayfun(@(i) find(A(:, i)), 1:n, uniformoutput=false); % neighbors
    K_nrm = sqrt(gamma_s) * full(sum(A, 2));                    % degree
    Km_nrm = accumarray(M, K_nrm, [max(M), 1]);                 % module degree
    Bd = full(diag(A)) - (K_nrm.^2);                            % self modularities
    Q1 = sum(Bd) / s;                                           % total modularity
    if Q1 - Q0 <= Args.tolerance
        break;
    else
        Q0 = Q1;
    end

    allneg_Bd = all(Bd < 0);
    flag = true;                                                % flag for within-hierarchy search
    while flag
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
                if allneg_Bd                                    % get module of neighbors
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
                if max_dQ > Args.tolerance                      % if maximal increase is positive
                    M(i) = v;                                   % reassign module
                    Km_nrm(v) = Km_nrm(v) + K_nrm(i);
                    Km_nrm(u) = Km_nrm(u) - K_nrm(i);

                    flag = true;                                % If we move the node to a different community, we add
                    Pi = Ji(M(Ji) ~= v).';                      % to the rear of the queue all neighbours of the node
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
    A = full(L.' * A * L);                                      % node-to-module strength
    n = length(A);                                              % number of nodes
    M = (1:n).';                                                % initial modules
end

end
