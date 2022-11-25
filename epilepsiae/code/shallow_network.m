function net = shallow_network(P, T, sizeHidden, noHiddenLayers, ew_base)
    % p should be inputsize * batch size
    % t should be outputsize * batch size
    % weight props is a line vector
    net = feedforwardnet(repmat(sizeHidden, 1, noHiddenLayers), "traingd");
    % no_instances = size(P,2);
    
    oht = onehotencode(T', 2)';
    net = configure(net, P, oht);
    net.layers{noHiddenLayers+1}.transferFcn = "softmax";
    net.layers{1:noHiddenLayers-1}.transferFcn = "purelin";
    net.layers{noHiddenLayers}.transferFcn = "logsig";
    % get_error_weights(T')
    % ew_base = [];

    % if size(weight_props,1) ~=0
    %     ew_base = get_error_weights(T').* weight_props';
    % else
    %     ew_base = 1;
    % end

    ew = ew_base(T,1);
    % size_train = size(P, 1);
    ew = ew_base(T);% zeros(size_train, 1);
    net.trainFcn = "traingdx";
    net.trainParam.max_fail = 100;
    % net.trainParam.min_grad = 0;
    net.trainParam.lr = 1e-1;
    size(P)
    size(oht)
    net = train(net, P, oht, [], [], ew);% , [], [], []);
end