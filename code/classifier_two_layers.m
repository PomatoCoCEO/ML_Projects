function net = classifier_two_layers(P,T, actFuncStr)
    trainFcn = "trainr";
    if actFuncStr == "hardlim"
        trainFcn = "trainc";
    end
    hiddenLayerSize = 256;
    net = fitnet(hiddenLayerSize, trainFcn);
    net = configure(net, P, T);
    net.layers{1}.transferFcn = actFuncStr;
    net.adaptFcn = "learngd"; % gradient method by default
    if actFuncStr == "hardlim"
        net.adaptFcn="learnp"; % perceptron weight and bias learning function
    end
    st = size(T);
    sp = size(P);

    
    % net.IW{1,1} = rand(hiddenLayerSize, sp(1)); % rsandom weight initialization
    % net.b{1,1} = rand(hiddenLayerSize, 1); % random bias initialization
    net.trainParam.epochs = 100;
    net.trainParam.goal = 1e-6;
    net.trainParam.lr = 0.1; 
    %view(net);
    net = train(net, P,T);
end