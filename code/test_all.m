function test_all()
    network_files={"1layer_C_hardlim.mat","1layer_C_logsig.mat","1layer_C_purelin.mat","1layer_hardlim.mat",
                "1layer_logsig.mat","1layer_purelin.mat","2layer_hardlim.mat","2layer_logsig.mat","2layer_purelin.mat",
                "Pattern.mat","perceptron.mat"}
    load("../data/AFW.mat") % associative memory
    load("P.mat") % load test data
    test_input=P;
    test_lbls = test_labels(); % load labels for test data
    % test_lbls = ones(size(test_lbls)); % depends on the 
    test_lbls_bin = binary_transform(test_lbls)';

    for i = 1:3
        file_name = "../data/"+network_files{i};
        load(file_name);
        real_output = sim(net, assoc * test_input); % these classifiers work with the output of the associative memory
        [c,cm,ind,per] = confusion(test_lbls_bin, real_output);
        fprintf("Classifier from %s after filter:\n" , file_name);
        fprintf("Accuracy: %f\n", (1-c));
        fprintf("Confusion matrix: \n");
        disp(cm);
        fprintf("\n");
    end

    for i=4:11
        file_name = "../data/"+network_files{i};
        load(file_name);
        real_output = sim(net, test_input); % these classifiers work with the output of the associative memory
        [c,cm,ind,per] = confusion(test_lbls_bin, real_output);
        fprintf("Classifier from %s\n" , file_name);
        fprintf("Accuracy: %f\n", (1-c));
        fprintf("Confusion matrix: \n");
        disp(cm);
        fprintf("\n");
    end

    
end

