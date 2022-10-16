function train_all()
    load("../data/p1000.mat") % load training data
    load("../data/test_input.mat") % load test data
    load("../data/labels_bin_1000.mat") % load labels for training data
    load("../data/target_1000.mat") % load Arial digits for filter
    test_input=P;
    test_lbls = test_labels(); % load labels for test data
    test_lbls = ones(size(test_lbls)); % depends on the 
    test_lbls_bin = binary_transform(test_lbls)';
    act_funct_strs = {'hardlim', 'purelin', 'logsig'}; % activation function strings

    % classifier 1: associative memory filter + classifier
    assoc = associative_memory_filter(p1000, target_1000);
    disp("after assoc");
    output_filter = assoc * p1000;

    net = binary_perceptron_filter(p1000, target_1000);
    % disp("after perceptron");
    save("../data/perceptron.mat", "net");
    output_filter_perceptron = sim(net, p1000);
    %! to be put in action again
    for i = 1:length(act_funct_strs)
        act_funct_str = act_funct_strs{i};
        net = classifier_one_layer(output_filter, labels_bin_1000, act_funct_strs{i}, true);
        save("../data/1layer_C_"+act_funct_str+"_no_softmax.mat", "net");
        disp("saved with the softmax");
        net = classifier_one_layer(output_filter, labels_bin_1000, act_funct_strs{i}, false);
        save("../data/1layer_C_"+act_funct_str+"_softmax.mat", "net");
    end
    %! do not mix softmax with hardlim function; it does not make sense

    %! to be put in action again
    for i = 1:length(act_funct_strs)
        act_funct_str = act_funct_strs{i};
        net = classifier_one_layer(output_filter_perceptron, labels_bin_1000, act_funct_strs{i}, true);
        save("../data/perceptron1C_"+act_funct_str+"_no_softmax.mat", "net");
        disp("saved with the softmax");
        net = classifier_one_layer(output_filter_perceptron, labels_bin_1000, act_funct_strs{i}, false);
        save("../data/perceptron1C_"+act_funct_str+"_softmax.mat", "net");
    end

    %classifier 2: no filtering; one layer
    for i = 1:length(act_funct_strs)
       act_funct_str = act_funct_strs{i};
       net = classifier_one_layer(p1000, labels_bin_1000, act_funct_str, false); %! working
       save("../data/1layer_"+act_funct_str+"_no_softmax.mat", "net");
        net = classifier_one_layer(p1000, labels_bin_1000, act_funct_str, true); %! working
        save("../data/1layer_"+act_funct_str+"_softmax.mat", "net");
    end
    % classifier 3: no filtering; two layers
    for i = 1:length(act_funct_strs)
        act_funct_str = act_funct_strs{i};
        net = classifier_two_layers(p1000, labels_bin_1000, act_funct_str, false);        
        save("../data/2layer_"+act_funct_str+"_no_softmax.mat", "net");
        net = classifier_two_layers(p1000, labels_bin_1000, act_funct_str,true);        
        save("../data/2layer_"+act_funct_str+"_softmax.mat", "net");
    end
    classifier 4: using patternnet
    net = pattern_net(p1000, labels_bin_1000);
    save("../data/pattern.mat", "net");

end