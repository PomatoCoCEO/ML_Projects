function net = binary_perceptron_filter(P ,T)
    net = perceptron; % Create a perceptron
    net = configure(net,P,T); % Configure the perceptron
    net.trainParam.goal = 1e-6; % Set the performance goal
    net.adaptFcn='learnp'; % Set the learning function
    net.trainFcn = "trainc"; % Set the training function
    net.trainParam.epochs = 1000; % Set the number of epochs
    net.performFcn='mse'; % Set the performance function
    net=train(net,P,T); % Train the perceptron
    
end