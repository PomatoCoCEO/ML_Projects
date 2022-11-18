function test_lstm(net, xtest, ytest)


%mini batch size used in training
YPred = classify(net,xtest,'MiniBatchSize',1024);

plotconfusion(ytest,YPred);

end