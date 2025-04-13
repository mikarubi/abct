classdef test_options < matlab.unittest.TestCase
    % Comprehensive Loyvain options tests

    methods (TestClassSetup)
        function intro_test(~)
            clc
            disp("*** Uncomment two snippets in step4_run to test" ...
                + newline + ...
                "     updates of objective and variable values.   ***.")
            pause(1)
        end
    end

    properties (TestParameter)
        Method = {"loyvain", "coloyvain"}
        NumClusters = {"one", "some", "all"};
        NumBatches = {"one", "some", "all"};
        Objective = {"kmodularity", "kmeans", "spectral"};
        Similarity = {"network", "corr", "cosim", "cov", "dot"};
        Start = {"greedy", "balanced", "random", "custom"}
        MaxIter = {1, 10};
        Replicates = {1, 10};
    end

    % Cycle through all combinations of options
    methods (Test)
        function loyvain_tests(TestCase, Method, NumClusters, NumBatches, ...
                Objective, Similarity, Start, MaxIter, Replicates)

            warning off
            rng(1)
            n = 16;
            q = 32;
            r = 24;
            X = randn(n, q);
            Y = randn(n, r);
            N = struct(one=1, some=n/4, all=n);
            k = N.(NumClusters);


            switch Method
                case "loyvain"
                    if Similarity == "network"
                        Data = max(corr(X'), 0);
                    elseif Objective == "spectral"
                        % ensures non-negative similarities
                        Data = (X + 4*mean(X))/5;
                    else
                        Data = X;
                    end
                case "coloyvain"
                    if Similarity == "network"
                        Data = max(corr(X, Y), 0);
                    elseif ismember(Objective, ["kmodularity", "spectral"])
                        % ensures non-negative similarities
                        X = (X + 9*mean(X, 2))/10;
                        Y = (9*X(:, 1:r) + randn(size(Y)))/10;
                    end
            end

            if Start == "custom"
                if Method == "loyvain"
                    Start = randi(k, n, 1);
                    Start(1:k) = randperm(k);
                    if rand < 0.5
                        k = 0;
                    end
                else
                    return;
                end
            end

            if Method == "loyvain"
                [M, Q] = loyvain(Data, k, ...
                    Objective, Similarity, ...
                    NumBatches=N.(NumBatches), ...
                    MaxIter=MaxIter, ...
                    Replicates=Replicates, ...
                    Start=Start, ...
                    Display="replicate");
            elseif Method == "coloyvain"
                if Similarity == "network"
                    Input = {Data};
                else
                    Input = {X, Y};
                end
                [M, Q] = coloyvain(Input{:}, k, ...
                    Objective, Similarity, ...
                    NumBatches=N.(NumBatches), ...
                    MaxIter=MaxIter, ...
                    Replicates=Replicates, ...
                    Start=Start, ...
                    Display="replicate");
            end

            TestCase.verifyThat(M, matlab.unittest.constraints.IsFinite);
            TestCase.verifyThat(Q, matlab.unittest.constraints.IsFinite);
            TestCase.verifyEqual(unique(M), 1:N.(NumClusters));
        end
    end
end
