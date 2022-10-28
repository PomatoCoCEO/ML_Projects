function stackednet = feature_autoencoder(P,T, hiddenLayerSizes)

    X = P;
    autoencs = [];
    for i = 1:length(hiddenLayerSizes)
        autoenc = trainAutoencoder(X,hiddenLayerSizes(i), ...
            'MaxEpochs',100, ...
            'L2WeightRegularization',0.004, ...
            'SparsityRegularization',4, ...
            'SparsityProportion',0.15, ...
            'ScaleData', false);
        autoencs = [autoencs autoenc];
        X = encode(autoenc,X);
    end

    softnet = trainSoftmaxLayer(X,T,'MaxEpochs',100);
    
    autoencs = mat2cell(autoencs,1,ones(1,numel(autoencs)));
    stackednet = stack(autoencs{:}, softnet );
end