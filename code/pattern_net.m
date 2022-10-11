function net = pattern_net(P,T)
    net= patternnet(10);
    net = configure(net, P, T);
    %view(net);

    net.trainParam.epochs = 100;
    net.trainParam.goal = 1e-6;
    net.trainParam.lr = 0.1; 

    net = train(net, P,T);
    % testing. Should move 
    % y = net(P)';
    % yy= convert_output(y);
    
    % y_target= repmat([10 1:9], 1,5)';
end