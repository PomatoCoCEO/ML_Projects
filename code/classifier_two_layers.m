function net = classifier_two_layers(P,T, actFuncStr,addSoftMax)
    trainFcn = "trainscg";
    adaptFunc = "learngd";
    if actFuncStr == "hardlim"
        trainFcn = "trainc";
        adaptFunc = "learnp";
    end
    hiddenLayerSize = 256;
    net = feedforwardnet([hiddenLayerSize], trainFcn);
    net.trainFcn = trainFcn;
    net.adaptFcn = adaptFunc;
    net.name="Two Layer Classifier "+actFuncStr;
    if addSoftMax
        str=net.trainFcn;
        net = feedforwardnet([hiddenLayerSize, 10], trainFcn);
        net.name="Two Layer Classifier "+actFuncStr+" with SoftMax";
        net.trainFcn = str;
        net.layers{3}.transferFcn = 'softmax';
        net.layers{2}.transferFcn = "purelin";
    end
    net.layers{1}.transferFcn = actFuncStr;
    net.trainParam.max_fail = 1000;
    net = configure(net, P, T);
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-6;
    net.trainParam.lr = 0.1;
    %view(net);
    net = train(net, P,T);
end