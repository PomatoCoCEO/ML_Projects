function net = lstm_network(P,T, numHiddenUnits, noFeatures, w)

    % t is categorical
    % p
    layers = [
        sequenceInputLayer(noFeatures),
        lstmLayer(numHiddenUnits, "OutputMode","last"),
        fullyConnectedLayer(4),
        softmaxLayer,
        classificationLayer(Classes = categorical([1,2,3,4]), ClassWeights= w)
    ];
    options=trainingOptions ("sgdm","MaxEpochs",150, "Shuffle","never", "Verbose",true, "ExecutionEnvironment","gpu", ...
            "MiniBatch", 1024);
    net = trainNetwork(P, T, layers, options);
end