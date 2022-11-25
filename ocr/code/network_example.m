function net = network_example()
net = network(1,2,[1;0],[1; 0],[0 0; 1 0],[0 1]);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'logsig';
net.layers{1}.dimensions = 2;
% net.inputs{1}.range = [0 1; -1 1];
% net.outputs{2}.range = [0 10; 0 10];

p = rand(20, 200)*20; % [0.5 0.9; -0.1 0.60];
ans= rand(1, 200)*4; % [0.56 2.5];
net = configure(net,p,ans);

net.IW{1,1} = rand(2,20);
net.b{1} = rand(2,1);
net.LW{2,1} = rand(1,2);
net.adaptFcn = "learnp";
net.trainFcn = 'trainc';
net.divideFcn = "dividerand";
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;
net = init(net);
net.trainParam.epochs = 30;
fprintf("Input weights: \n");
disp(net.IW{1,1});
fprintf("Biases: ");
disp(net.b{1});
fprintf("Layer weights: \n");
disp(net.LW{2,1});
y = sim(net,p);
net = train(net, p, ans);
y = sim(net,p);
% fprintf("Layers1: \n");
% disp(net.layers{1});
% fprintf("Layers2: \n");
% disp(net.layers{2});
% fprintf("Input weights: \n");
% disp(net.IW{1,1});
fprintf("Biases: ");
disp(net.b{1});
fprintf("Layer weights: \n");
disp(net.LW{2,1});

end
