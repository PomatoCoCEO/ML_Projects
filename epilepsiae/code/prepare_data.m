function [FeatVectSel, Trg , classif, classif_cat,  images,  classif_images, cell_input] = prepare_data()

    load("../data/44202.mat", "FeatVectSel", "Trg");
    % normal parametres
    classif = make_classification(Trg)';
    classif_cat = onehotdecode(classif, [1 2 3 4], 2);
    % preparing the images for the CNN classification
    [images, classif_images] = prepare_images(FeatVectSel, classif);
    cell_input = num2cell(FeatVectSel', 1); % turns the input array into a cell array
    % classif_cat = categorical(classif_scalar);
    
end

