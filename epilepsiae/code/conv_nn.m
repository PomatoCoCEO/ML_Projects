function net = conv_nn(P,T,validation_data)
    % 
    % P is a cell array in which each element is an image, and it is
    % according to the matlab standards, 29*29*1*no_of_images
    % T is a categorical array representing the classes
    % it has dimensions no_of_images*1 :)
    no_features = size(P,1);
    no_classes = 4;
    weights = get_error_weights(T).*[1;5;1;1];
    classes = categorical([1 2 3 4]);

    layers = [
        imageInputLayer([no_features,no_features,1])
        convolution2dLayer(4, 3)
        reluLayer
        maxPooling2dLayer(2, "Stride", 2)
        fullyConnectedLayer(no_classes)
        softmaxLayer
        classificationLayer(Classes = classes, ClassWeights=weights)
    ];
    disp(size(P))
    options = trainingOptions("adam", ...
        "Plots","training-progress","Verbose", true, "ValidationData",validation_data); 
    %    ,true,'ValidationData',validation_data);
    net = trainNetwork(P, T, layers, options);
end