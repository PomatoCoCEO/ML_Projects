function [dataTrain, dataTest, trgTrain, trgTest ] = train_test_split_images(data, trg, testProportion, shuffle)
% data is a width * height * 1 * no_of_images
% sensitivity + specificity
trainEndCoef = floor((1 - testProportion) * size(data,4));
if shuffle
    sz = size(data,4);
    rp = randperm(sz);
    dataTrain = data(:,:,:,rp(1:trainEndCoef));
    dataTest = data(:,:,:,rp(trainEndCoef+1:end));
    trgTrain = trg(rp(1:trainEndCoef),:);
    trgTest = trg(rp(trainEndCoef+1:end),:);
else 
    dataTrain = data(:,:,:,1:trainEndCoef);
    dataTest = data(:,:,:,trainEndCoef+1:end);
    trgTrain = trg(1:trainEndCoef, :);
    trgTest = trg(trainEndCoef+1:end,:);
end

end