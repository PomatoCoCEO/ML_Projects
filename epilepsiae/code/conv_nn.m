function net = conv_nn(P,T)
    % P is a cell array in which each element is an image, and it is
    % according to the matlab standards, 29*29*1*no_of_images
    % T is a categorical array representing the classes
    % it has dimensions no_of_images*1
    no_features = size(P,1);
    no_classes = size(T,2);
    layers = [
        imageInputLayer([no_features,no_features,1])
        convolution2dLayer(4, 3)
        reluLayer
        maxPooling2dLayer(2, "Stride", 2)
        fullyConnectedLayer(no_classes)
        softmaxLayer
        classificationLayer
    ];
    target_scalar = categorical(binary_to_scalar(T))';
    disp(size(target_scalar))
    disp(size(P))
    options = trainingOptions("sgdm");
    net = trainNetwork(P, target_scalar, layers, options);
    % to test network: 
    % YPred = classify(net,imdsTest);
    % YTest = imdsTest.Labels;
    % accuracy = sum(YPred == YTest)/numel(YTest)
    % we also need the confusion matrix for more valid conclusions
end