function net = pattern_net(P,T)
    net= patternnet(10);
    net = configure(net, P, T);
    % net = addLayers(net, softmaxLayer);

    net.trainParam.epochs = 100;
    net.trainParam.goal = 1e-6;
    net.trainParam.lr = 0.1; 
    % view(net);
    net = train(net, P,T);
    % this function already has a softmax layer, so there is no need to try to add one
end