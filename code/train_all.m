function train_all()
    load("../data/P_total.mat") % load training data
    load("../data/test_input.mat") % load test data
    load("../data/labels_binary_total.mat") % load labels for training data
    load("../data/target_filter.mat") % load Arial digits for filter
    test_lbls = test_labels(); % load labels for test data
    act_funct_strs = {'hardlim', 'purelin', 'logsig'}; % activation function strings

    %% classifier 1: associative memory filter + classifier
    assoc = associative_memory_filter(P_total, target_filter);
    output_filter = assoc * P_total;
    for i = 1:length(act_funct_strs)
        act_funct_str = act_funct_strs{i};
        one_layer_classifier = classifier_one_layer(output_filter, labels_binary_total, act_funct_strs{i});
        real_outputs = sim(one_layer_classifier, assoc * test_input);
        % calculate the accuracy
        % convert output for classification evaluation: use convert_output
        [c,cm,ind,per] = confusion(labels_binary_total, real_outputs);
        print("Classifier with associative memory filter and one layer classifier with activation function " + act_funct_str);
        print("Accuracy: ", 1-c);
        print("Confusion matrix: ", cm);
        % possibility of using a softmax layer
    end

    % classifier 2: no filtering; one layer
    for i = 1:length(act_funct_strs)
        act_funct_str = act_funct_strs{i};
        one_layer = classifier_one_layer(P_total, labels_binary_total, act_funct_str);
        real_outputs = sim(one_layer, P_total);
        % convert output for classification evaluation: use convert_output
        [c,cm,ind,per] = confusion(labels_binary_total, real_outputs);
        print("Classifier with one layer and activation function " + act_funct_str)
        print("Accuracy: ", 1-c)
        print("Confusion matrix: ", cm)
    end

    % classifier 3: no filtering; two layers
    for i = 1:length(act_funct_strs)
        act_funct_str = act_funct_strs{i};
        one_layer = classifier_two_layers(P_total, labels_binary_total, act_funct_str);
        real_outputs = sim(one_layer, P_total);
        % convert output for classification evaluation: use convert_output
        [c,cm,ind,per] = confusion(labels_binary_total, real_outputs);
        print("Classifier with one layer and activation function " + act_funct_str)
        print("Accuracy: ", 1-c)
        print("Confusion matrix: ", cm)
    end

    % classifier 4: using patternnet

    pn = pattern_net(P_total, labels_binary_total);
    real_outputs_pn = sim(pn, P_total);
    [c_pn,cm_pn,ind_pn,per_pn] = confusion(labels_binary_total, real_outputs);
    print("Classifier with one layer and activation function " + act_funct_str)
    print("Accuracy: ", 1-c_pn)
    print("Confusion matrix: ", cm_pn)

end