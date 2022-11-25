function net = train_rec(P, T, T_ohe ,maxDelay, hiddenSizes, hiddenNo, trainFcn, ew_base)

    net = layrecnet(1:maxDelay, repmat(hiddenSizes, 1, hiddenNo), trainFcn);
    
    net = configure(net, P', T_ohe');
    % ew_base = [20 20 1 1];
    %ew_base = get_error_weights(T);
    %view(net);
    ew = ew_base(T,1);
    
    net.trainParam.max_fail = 100;
    net.trainParam.min_grad = 0;
    net.trainFcn = "traingdx";
    net.trainParam.lr = 1e-4;
    %size(T_ohe)
    %size( P)
    %size(ew)
    %net.inputs{1}.size
    net = train(net, P', T_ohe', [], [], ew);
end