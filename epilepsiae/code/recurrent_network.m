function net = recurrent_network(P,T, maxDelay, hiddenSizes, hiddenNo, trainFcn)
    net = layrecnet(1:maxDelay, repmat(hiddenSizes, 1, hiddenNo));
    net.trainFcn = trainFcn;
    size(T)
    size(P)
    
    view(net);
    ew_base = get_error_weights(T').* [1/2 1/3 1/2 1];
    ew = ew_base(T);
    [Xs,Xi,Ai,Ts,EWs,~] = preparets(net,P,T,[], ew);
    net = configure(net, Xs,Ts);
    net = train(net, Xs, Ts, Xi, Ai, EWs);
end