function net = train_lstm_cb(ratio)

[FeatVectSel, Trg , classif, classif_cat,  images,  classif_images, cell_input] = prepare_data();


[dataTrain, dataTest, trgTrain_cat, trgTest_cat ] = train_test_split_nocell(FeatVectSel, classif_cat, 0.3, false);
 [dataTrain, dataTest, trgTrain, trgTest ] = train_test_split_nocell(FeatVectSel, classif, 0.3, false);

[x_sub, t_sub , t_sub_ohe] = class_balancing(dataTrain, trgTrain_cat, trgTrain, ratio);

dataTrainCell =num2cell(x_sub', 1);

net = lstm_network(dataTrainCell, t_sub, 10,29);

dataTestCell =num2cell(dataTest', 1);
y_pred = classify(net, dataTestCell);
y_true = trgTest_cat;
[sensitivity_2, specificity_2] = report_spec_sens(y_true, y_pred, 2);
[sensitivity_3, specificity_3] = report_spec_sens(y_true, y_pred, 3);


end