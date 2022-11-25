function test_all(folderName)
    networkFolderPath = "../data/"+folderName;
    files=ls("../data/"+folderName+"/*.mat");
    load("../data/P_test.mat"); % load test data
    test_input=P_test;
    test_lbls = test_labels(); % load labels for test data
    test_lbls_bin = binary_transform(test_lbls)';
    
    %load("../data/labels_bin_1000")
    %test_lbls_bin = labels_bin_1000;
    % used to test training
    
    load("../data/"+folderName+"/AFW.mat"); % associative layer
    load("../data/"+folderName+"/perceptron.mat"); % associative layer
    perceptron = net;
    sf = size(files)
    for i = 1:sf(1)
        file=char(strtrim(files(i,:))); % falta fazer loading e avaliar a accuracy e etc,
        load("../data/"+folderName+"/"+file);
        fprintf("file name: %s \n" , file);
        c=1; cm=1; ind=1; per=1; real_output = [];
        if strcmp(file, "AFW.mat") || strcmp(file, "perceptron.mat")
            continue
        end
        % tendo em conta os nomes
        if strcmp(file(1:9),'1layer_C_')
            output_assoc = weights * test_input;
            real_output = sim(net, output_assoc);
        elseif strcmp(file(1:10),'perceptron')
            % associative layer and perceptron filter
            output_assoc = sim(perceptron, test_input);
            real_output = sim(net, output_assoc);
        elseif strcmp(file(1:7),"1layer_") || strcmp(file(1:7), "2layer_") || strcmp(file,"pattern.mat") 
            % 1 and 2 layer classifiers
            real_output = sim(net, test_input);
        end
        [c, cm, ind, per] = confusion(test_lbls_bin, real_output);
        fprintf("Classifier from %s:\n" , net.name);
        fprintf("Accuracy: %f\n", (1-c));
        %fprintf("Confusion matrix: \n");
        %disp(cm);
        fprintf("\n");
    end
end
