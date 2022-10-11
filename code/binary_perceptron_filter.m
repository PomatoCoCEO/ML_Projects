function net = binary_perceptron_filter(P ,T)
    net = perceptron; % Create a perceptron
    net = configure(net,P,T); % Configure the perceptron
    net.trainParam.epochs = 100; % Set the number of epochs
    net.trainParam.goal = 1e-6; % Set the performance goal
    net.adaptFcn='learnp'; % Set the learning function
    net.performFcn='mse'; % Set the performance function
    %view(net)
    net=train(net,P,T); % Train the perceptron
    % num_inputs = 2;
    % num_outputs = 3;
    % input_connect_filter = [ones(1,num_inputs)];

    % net = network; % we will start with the filter: 2.1 a)
    % net.numLayers = 1;
    % net.numInputs = num_inputs;
    % net.biasConnect = [ true]; % binary perceptron
    % net.outputConnect = [true];
    % net.inputConnect = input_connect_filter;
    % net.trainFcn = 'trainc'; % using the pseudo-inverse method
    % net.adaptFcn="learnp";
    % net.layers{1}.transferFcn = 'hardlim'; % heaviside activation function 
    % net.layers{1}.size = num_outputs;
    % net.b{1} = rand(num_outputs,1);
    % net.outputs{1} = zeros(1,num_outputs);
    % % net.IW{1,1} = rand(256,256);
    % view(net)
    % net = train(net, P, T); % training the network
end