function net = train_lstm(xtrain, ytrain)

% assumes xtrain is a cell array
%no transposes
net = lstm_network(xtrain, ytrain, 5);

end