function net = train_rec_cb(ratio)

[FeatVectSel, Trg , classif, classif_cat,  images,  classif_images, cell_input] = prepare_data();


[dataTrain, dataTest, trgTrain_cat, trgTest_cat ] = train_test_split_nocell(FeatVectSel, classif_cat, 0.3, false);
 [dataTrain, dataTest, trgTrain, trgTest ] = train_test_split_nocell(FeatVectSel, classif, 0.3, false);

[x_sub, t_sub , t_sub_ohe] = class_balancing(dataTrain, trgTrain_cat, trgTrain, ratio);


net = train_rec(x_sub, t_sub , t_sub_ohe , 10,29,5,'traingdx');

y_pred = net(dataTest');
plotconfusion(trgTest' , y_pred);

end