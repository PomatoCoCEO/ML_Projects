function net = lstm_network(P,T, validation_data, numHiddenUnits, weight_proportions)
% weight proportions is a 4-element  array
%validation data is a cell array in which the first element is the 
%test input cell array and the second is the test output categorical array
    weights = get_error_weights(T');
    if size(weight_proportions,1) ~= 0
        weights = weights .* weight_proportions';
    end
    size(weights)
    class_labels = categorical([1 2 3 4]);
    layers = [
        sequenceInputLayer(29),
        lstmLayer(numHiddenUnits, "OutputMode","last"),
        fullyConnectedLayer(4),
        softmaxLayer,
        classificationLayer(Classes = class_labels, ClassWeights = weights)
    ];
    options=trainingOptions ("adam","MaxEpochs",150, "Shuffle","never", "Verbose",true, ...
            "MiniBatch", 1024, "ValidationData", validation_data, "InitialLearnRate", 1e-5, ...
            "Plots", "training-progress");
    net = trainNetwork(P, T, layers, options);
end