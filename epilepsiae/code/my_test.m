function my_test(NN_name, EW, CB , dataset_name, train, output)

% EW, autoencoder and balance are bool values
% predict is a bool value. If false mean detect
NN_name
try
    load("../data/" + dataset_name);
catch ME
    s = "File does not exist. Try again!";
    
    output.Value= s;

    return; 
end
sensitivity_2 =0;
sensitivity_3 =0;
specificity_2 =0;
specificity_3 =0;

classif = make_classification(Trg)';
classif_cat = onehotdecode(classif, [1 2 3 4], 2);
% preparing the images for the CNN classification
[images, classif_images] = prepare_images(FeatVectSel, classif);
cell_input = num2cell(FeatVectSel', 1);


[~, ~, trgTrain, trgTest ] = train_test_split_nocell(FeatVectSel, classif, 0.3, false);
[dataTrain, dataTest, trgTrain_cat, trgTest_cat ] = train_test_split_nocell(FeatVectSel, classif_cat, 0.3, false);
[imagesTrain, imagesTest, trgImTrain, trgImTest] = train_test_split_images(images, classif_images, 0.3, false);

%load differently trained networks
name= NN_name;
    
    if EW
        name = name+"_EW";
    end
    if CB 
        name= name + "_CB";
    end
if train

    %train new net and test it
    w= [1,1,1,1];
    if NN_name=="CNN"
        if EW
            w=get_error_weights(trgTrain_cat);
        end
        [net,s] = train_network("CNN",name+"_new", imagesTrain, trgImTrain, {imagesTest, trgImTest}, w );
    elseif NN_name == "LSTM"
        if EW
            w=get_error_weights(trgTrain_cat);
        end
        if CB
            [x_sub, t_sub , t_sub_ohe] = class_balancing(dataTrain, trgTrain_cat, trgTrain, 0.5);
            dataTrainCell =num2cell(x_sub', 1);
            dataTestCell =num2cell(dataTest', 1);
             [net,s] = train_network("LSTM", name+"_new", dataTrainCell, t_sub, {dataTestCell, trgTest_cat}, 5 , w);
        else
            dataTrainCell =num2cell(dataTrain', 1);
            dataTestCell =num2cell(dataTest', 1);
            [net,s] = train_network("LSTM", name+"_new", dataTrainCell, trgTrain, {dataTestCell, trgTest_cat},5, w);
        end


    elseif NN_name == "SNN"
        if EW
            w=get_error_weights(trgTrain_cat);
        end
        if CB
            [x_sub, t_sub , t_sub_ohe] = class_balancing(dataTrain, trgTrain_cat, trgTrain, 0.5);

            [net,s] = train_network("SNN", name+"_new", x_sub, t_sub_ohe, {dataTest, trgTest},  3, w);
        else

            [net,s] = train_network("SNN", name+"_new", dataTrain, trgTrain, {dataTest, trgTest}, 3, w);
        end

    elseif NN_name == "RNN"
        if EW
            w=get_error_weights(trgTrain_cat);
        end

        if CB
            [x_sub, t_sub , t_sub_ohe] = class_balancing(dataTrain, trgTrain_cat, trgTrain, 0.5);

            [net,s] = train_network("RNN", name+"_new", x_sub, t_sub_ohe, {dataTest, trgTest},  t_sub, w);
        else

            [net,s] = train_network("RNN", name+"_new", dataTrain, trgTrain, {dataTest, trgTest}, trgTrain_cat, w);
        end
    end 


    output.Value = s;
else
    %only test
    
    

    
    
    if NN_name=="CNN"

        load("../data/nets/"+ name);

        result = classify(net, imagesTest);
        y_true = onehotdecode(trgImTest, [1,2,3,4], 2);
        size(result)
        size(y_true)
        
        [sensitivity_2, specificity_2] = report_spec_sens(y_true, result, 2);
        [sensitivity_3, specificity_3] = report_spec_sens(y_true, result, 3);
    
    elseif NN_name == "LSTM"
        

        dataTestCell =num2cell(dataTest', 1);

        y_true = trgTest_cat;

        if EW && CB
        %different best results
        p_name = name+"_pred";
        d_name = name+"_det";
        load("../data/nets/"+ p_name);
        
        y_pred_p = classify(net, dataTestCell);

        load("../data/nets/"+ d_name);
        
        y_pred_d = classify(net, dataTestCell);

        

        [sensitivity_2, specificity_2] = report_spec_sens(y_true, y_pred_p, 2);
        [sensitivity_3, specificity_3] = report_spec_sens(y_true, y_pred_d, 3);
        else
            load("../data/nets/"+ name);
            y_pred = classify(net, dataTestCell);

            [sensitivity_2, specificity_2] = report_spec_sens(y_true, y_pred, 2);
            [sensitivity_3, specificity_3] = report_spec_sens(y_true, y_pred, 3);
        
        end

  
    elseif NN_name == "SNN"
    
        
        load("../data/nets/"+ name);
        % load previously train model
        % name should be net
    
        y_pred = net(dataTest');
        y_true = trgTest';
        size(y_pred)
        size(y_true)
        
        [sensitivity_2, specificity_2] = report_spec_sens_ohe(y_true, y_pred, 2);
        [sensitivity_3, specificity_3] = report_spec_sens_ohe(y_true, y_pred, 3);
    elseif NN_name == "RNN"
        
        % load previously train model
        % name should be net
        load("../data/nets/"+ name);
        y_pred = net(dataTest');
        y_true = trgTest';

        [sensitivity_2, specificity_2] = report_spec_sens_ohe(y_true, y_pred, 2);
        [sensitivity_3, specificity_3] = report_spec_sens_ohe(y_true, y_pred, 3);
        
    
    end
    s2 = sprintf('Predict: Sensitivity = %.5f, Specificity: %.5f',sensitivity_2, specificity_2);
    s3 = sprintf('Detect: Sensitivity = %.5f, Specificity: %.5f',sensitivity_3, specificity_3);
    disp(s2)
    disp(s3)
    output.Value = string(s2) + newline+ string(s3);
end






end