function net = shallow_network(P, T, sizeHidden, noHiddenLayers)
    % p should be inputsize * batch size
    % t should be outputsize * batch size
    net = feedforwardnet(repmat(sizeHidden, 1, noHiddenLayers), "traingd");
    % no_instances = size(P,2);
    
    oht = onehotencode(T', 2)';
    net = configure(net, P, oht);
    net.layers{noHiddenLayers+1}.transferFcn = "softmax";
    net.layers{1:noHiddenLayers-1}.transferFcn = "purelin";
    net.layers{noHiddenLayers}.transferFcn = "logsig";
    get_error_weights(T')
    ew_base = get_error_weights(T').* [1/2 1/3 1/2 1];

    % size_train = size(P, 1);
    ew = ew_base(T);% zeros(size_train, 1);
    size(ew)
    % view(net);
    net.trainFcn = "traingdx";
    net.trainParam.max_fail = 100;
    % net.trainParam.min_grad = 0;
    net.trainParam.lr = 1e-1;
    net = train(net, P, oht, [], [], ew);% , [], [], []);
end