function net = classifier_two_layers(P,T, actFuncStr,addSoftMax)
    trainFcn = "trainr";
    if actFuncStr == "hardlim"
        trainFcn = "trainc";
    end
    hiddenLayerSize = 256;
    net = feedforwardnet([hiddenLayerSize], trainFcn);
    if addSoftMax
        str=net.trainFcn;
        net = feedforwardnet([hiddenLayerSize, 10], trainFcn);
        net.trainFcn = str;
        net.layers{3}.transferFcn = 'softmax';
        net.layers{2}.transferFcn = "purelin";
    end
    net = configure(net, P, T);
    net.layers{1}.transferFcn = actFuncStr;
    net.adaptFcn = "learngd"; % gradient method by default
    if actFuncStr == "hardlim"
        net.adaptFcn="learnp"; % perceptron weight and bias learning function
    end
    net = configure(net, P, T);
    st = size(T);
    sp = size(P);
    net.trainParam.epochs = 10;
    net.trainParam.goal = 1e-6;
    net.trainParam.lr = 0.1;
    view(net);
    net = train(net, P,T);
end