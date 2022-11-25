function net = train_lstm_encoder()
[FeatVectSel, Trg , classif, classif_cat,  images,  classif_images, cell_input] = prepare_data();
load('encoder.mat')
X = encode( encoder, FeatVectSel');

[dataTrain, dataTest, trgTrain_cat, trgTest_cat ] = train_test_split_nocell( X' , classif_cat, 0.3, false);
dataTrainCell =num2cell(dataTrain', 1);
net = lstm_network(dataTrainCell, trgTrain_cat, 10, 3);

dataTestCell =num2cell(dataTest', 1);
y_pred = classify(net, dataTestCell);
y_true = trgTest_cat;
[sensitivity_2, specificity_2] = report_spec_sens(y_true, y_pred, 2);
[sensitivity_3, specificity_3] = report_spec_sens(y_true, y_pred, 3);


end