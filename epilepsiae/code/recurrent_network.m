function net = recurrent_network(P,T, maxDelay, hiddenSizes, hiddenNo, trainFcn)
    net = layrecnet(1:maxDelay, repmat(hiddenSizes, 1, hiddenNo), trainFcn);
    net = configure(net, P,T);
    view(net);
    net = train(net, P, T);
end