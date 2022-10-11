function net = classifier_one_layer(P,T, actFuncStr)
    net = linearlayer;
    net = configure(net, P, T);
    net.layers{1}.transferFcn = actFuncStr;
    net.adaptFcn = "learngd"; % gradient method by default
    net.trainFcn = "trainr";
    if actFuncStr == "hardlim"
        net.adaptFcn="learnp"; % perceptron weight and bias learning function
        net.trainFcn = "trainc";
    end
    st = size(T);
    sp = size(P);
    net.IW{1,1} = rand(st(1), sp(1)); % random weight initialization
    net.b{1,1} = rand(st(1), 1); % random bias initialization
    net.trainParam.epochs = 100;
    net.trainParam.goal = 1e-6;
    net.trainParam.lr = 0.1; 
    %view(net);
    net = train(net, P,T);
end