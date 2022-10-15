function net = classifier_one_layer(P,T, actFuncStr, addSoftMax)
    net = linearlayer;
    % net = feedforwardnet([], "trainr");
    net = configure(net, P, T);
    net.layers{1}.transferFcn = actFuncStr;
    
    st = size(T);
    net.adaptFcn = "learngd"; % gradient method by default
    if actFuncStr == "hardlim"
        if ~addSoftMax
            net.adaptFcn="learnp"; % perceptron weight and bias learning function
            net.trainFcn = "trainc";
        else
            net.layers{1}.transferFcn = "softmax";
        end
    elseif addSoftMax
            % net = addLayers(net, softmaxLayer);
            str = net.trainFcn;
            net = feedforwardnet([st(1)],str);
            net.layers{1}.transferFcn = actFuncStr;
            net.layers{2}.transferFcn = "softmax"; % softmax layer without bias
            net.biasConnect(2)=false;
            view(net);
    end
    view(net);
    % sp = size(P)
    % net.IW{1,1} = rand(st(1), sp(1)); % random weight initialization
    % net.b{1,1} = rand(st(1), 1); % random bias initialization
    
    net = init(net);
    net = configure(net, P,T);
    net.trainParam.epochs = 10;
    net.trainParam.goal = 1e-6;
    net.trainParam.lr = 0.1; 
    % view(net);
    net = train(net, P,T);
end