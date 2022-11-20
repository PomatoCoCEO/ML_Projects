function test_by_start(str)
%TEST_LSTMS Summary of this function goes here
%   Detailed explanation goes here
    [FeatVectSel, Trg , classif, classif_cat,  images, classif_images, cell_input] = prepare_data();
    [dataTrain, dataTest, trgTrain, trgTest ] = train_test_split(cell_input, classif_cat, 0.3, false);
    files = split(ls('../data/'+str+'*.mat'));
    
    for i = 1:size(files, 1)-1
        file = files{i};
        load(file);
        dTest = dataTest;
        tTest = trgTest;
        if strcmp(str, "lstm")
            res = classify(net, dTest);
        elseif strcmp(str, "cnn")
            [imageTrain, imageTest, trgImageTrain, trgImageTest] = train_test_split_images(images, classif_images, 0.3, true);
            dTest = imageTest;
            tTest = trgImageTest;
            res = classify(net, dTest);
        elseif strcmp(str, "shallow")
            dte = cell2mat(dataTest)';
            res = onehotdecode(net(dte'), [1,2,3,4], 1)';
        end
        [sensitivity3, specificity3] = report_spec_sens(tTest, res,3);
        [sensitivity2, specificity2] = report_spec_sens(tTest, res,2);
        fprintf("%s: Prediction: (%f, %f); Detection: (%f, %f)\n", file, sensitivity2, specificity2, sensitivity3, specificity3);
    end
    
    
end