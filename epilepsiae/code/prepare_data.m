function [FeatVectSel, Trg , classif, classif_scalar] = prepare_data()

    load("../data/44202.mat");
    classif = make_classification(Trg);
    classif_scalar = binary_to_scalar(classif');


end

