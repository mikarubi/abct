classdef opts_tests < matlab.unittest.TestCase
    % Comprehensive Loyvain options tests

    properties (TestParameter)
        NumClusters = {"one", "some", "all"};
        NumBatches = {"one", "some", "all"};
        Objective = {"modularity", "kmeans", "spectral"};
        Similarity = {"network", "corr", "cosim", "cov", "dot"};
        Start = {"greedy", "balanced", "random", "custom"}
        MaxIter = {1, 10};
        Replicates = {1, 10};
    end

    % Cycle through all combinations of options
    methods (Test)
        function loyvain_tests(TestCase, NumClusters, NumBatches, ...
                Objective, Similarity, Start, MaxIter, Replicates)

            warning off
            rng(1)
            n = 16;
            t = 32;
            X = randn(n, t);
            N = struct(one=1, some=n/4, all=n);
            k = N.(NumClusters);
            if Similarity == "network"
                Data = max(corr(X'), 0);
            elseif Objective == "spectral"
                % ensures non-negative similarities
                Data = (X + 4*mean(X))/5;
            else
                Data = X;
            end
            if Start == "custom"
                Start = randi(k, n, 1);
                Start(1:k) = randperm(k);
                if rand < 0.5
                    k = 0;
                end
            end

            [M, Q] = loyvain(Data, k, Objective, ...
                Similarity=Similarity, ...
                NumBatches=N.(NumBatches), ...
                MaxIter=MaxIter, ...
                Replicates=Replicates, ...
                Start=Start, ...
                Display="replicate");

            TestCase.verifyThat(M, matlab.unittest.constraints.IsFinite);
            TestCase.verifyThat(Q, matlab.unittest.constraints.IsFinite);
        end
    end
end
