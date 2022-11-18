function [dataTrain, dataTest, trgTrain, trgTest ] = train_test_split(data, trg)

% sensitivity + specificity

% Cross varidation (train: 70%, test: 30%)
cv = cvpartition(size(data,1),'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
dataTrain = data(~idx,:);
dataTest  = data(idx,:);

trgTrain = trg(~idx,:);
trgTest = trg(idx,:);

end