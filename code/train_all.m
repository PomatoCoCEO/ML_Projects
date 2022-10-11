function train_all()
    load("../data/P_total.mat") % load training data
    load("P.mat") % load test data
    load("../data/labels_binary_total.mat") % load labels for training data
    load("../data/target_filter.mat") % load Arial digits for filter
    test_input=P;
    %test_lbls = test_labels(); % load labels for test data
    %test_lbls = ones(size(test_lbls)); % depends on the 
    %test_lbls_bin = binary_transform(test_lbls)';
    act_funct_strs = {'hardlim', 'purelin', 'logsig'}; % activation function strings

    %% classifier 1: associative memory filter + classifier
    assoc = associative_memory_filter(P_total, target_filter);
    
    output_filter = assoc * P_total;

    net = binary_perceptron_filter(P_total, labels_binary_total);

    save("../data/perceptron.mat", "net");
    for i = 1:length(act_funct_strs)
        act_funct_str = act_funct_strs{i};
        net = classifier_one_layer(output_filter, labels_binary_total, act_funct_strs{i});
        save("../data/1layer_C_"+act_funct_str+".mat", "net");
        % real_outputs = sim(one_layer_classifier, assoc * test_input);
        % % calculate the accuracy
        % % convert output for classification evaluation: use convert_output
        % % size(real_outputs)
        % [c,cm,ind,per] = confusion(test_lbls_bin, real_outputs);
        % fprintf("Classifier with associative memory filter and one layer classifier with activation function %s\n" , act_funct_str);
        % fprintf("Accuracy: %f\n", (1-c));
        % fprintf("Confusion matrix: \n");
        % disp(cm);
        % fprintf("\n");
        % possibility of using a softmax layer
    end

    % classifier 2: no filtering; one layer
    for i = 1:length(act_funct_strs)
        act_funct_str = act_funct_strs{i};
        net = classifier_one_layer(P_total, labels_binary_total, act_funct_str);
        save("../data/1layer_"+act_funct_str+".mat", "net");
    end
    % classifier 3: no filtering; two layers
    for i = 1:length(act_funct_strs)
        act_funct_str = act_funct_strs{i};
        net = classifier_two_layers(P_total, labels_binary_total, act_funct_str);        
        save("../data/2layer_"+act_funct_str+".mat", "net");
    end
    % classifier 4: using patternnet
    net = pattern_net(P_total, labels_binary_total);
    save("../data/pattern.mat", "net");

end