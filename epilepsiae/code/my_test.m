function my_test(NN_name, options , dataset_name, train, output)

% EW, autoencoder and balance are bool values
% predict is a bool value. If false mean detect

try
    load("../data/" + dataset_name);
catch ME
    s = "File does not exist. Try again!";
    
    output.Value= s;

    return; 
end

[FeatVectSel, Trg , classif, classif_cat,  images,  classif_images, cell_input] = prepare_data();
[dataTrain, dataTest, trgTrain, trgTest ] = train_test_split_nocell(FeatVectSel, classif, 0.3, false);
[dataTrain, dataTest, trgTrain_cat, trgTest_cat ] = train_test_split_nocell(FeatVectSel, classif_cat, 0.3, false);


%load differently trained networks

if train

    %train new net and test it
    
else
    %only test
    name= NN_name + options;
    
    
    disp(options);
    
    
    if NN_name=="CNN"
        p_name = name+"_pred";
        d_name = name+"_det";
        
    
    
    
    elseif NN_name == "LSTM"
        p_name = name+"_pred";
        d_name = name+"_det";
    
    
    
    
    elseif NN_name == "SNN"
    
        
    
        % load previously train model
        % name should be net
    
        y_pred = net(dataTest');
        y_true = trgTrain';
    
    elseif NN_name == "RNN"
        
        % load previously train model
        % name should be net
    
        y_pred = net(dataTest');
        y_true = trgTrain';
    
    
    end

end

[sensitivity_2, specificity_2] = report_spec_sens(y_true, y_pred, 2);
[sensitivity_3, specificity_3] = report_spec_sens(y_true, y_pred, 3);


s2 = sprintf('Predict: Sensitivity = %.2f, Specificity: %.2f',sensitivity_2, specificity_2);
s3 = sprintf('Detect: Sensitivity = %.2f, Specificity: %.2f',sensitivity_3, specificity_3);

output.Value = s2+ newline+ s3;

end