function net = classifier_one_layer(P,T, actFuncStr, addSoftMax)
    net = linearlayer;
    % net = feedforwardnet([], "trainr");
    net = configure(net, P, T);
    net.layers{1}.transferFcn = actFuncStr;
    net.name="One Layer Classifier "+actFuncStr;
    st = size(T);
    net.adaptFcn = "learngd"; % gradient method by default
    net.trainFcn = "trainr";
    if actFuncStr == "hardlim"
        if ~addSoftMax
            net.adaptFcn="learnp"; % perceptron weight and bias learning function
            net.trainFcn = "trainc";
        else
            net.name="One Layer Classifier "+actFuncStr+" with Softmax";
            net.layers{1}.transferFcn = "softmax";
        end
    elseif addSoftMax
            % net = addLayers(net, softmaxLayer);
            str = net.trainFcn;
            net = feedforwardnet([st(1)],str);
            net.name="One Layer Classifier "+actFuncStr+" with Softmax";
            net.adaptFcn = "learngd";
            net.layers{1}.transferFcn = actFuncStr;
            net.layers{2}.transferFcn = "softmax"; % softmax layer without bias
            net.biasConnect(2)=false;
    end
    net = init(net);
    net = configure(net, P,T);
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-6;
    net.trainParam.lr = 0.1; 
    %view(net);
    net = train(net, P,T);
end