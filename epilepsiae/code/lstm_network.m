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
    options=trainingOptions ("adam","MaxEpochs",150, "Shuffle","never", "Verbose",true, ...
            "MiniBatch", 1024, "ValidationData", validation_data, "InitialLearnRate", 1e-5, ...
            "Plots", "training-progress");
    net = trainNetwork(P, T, layers, options);
end