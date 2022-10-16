function train_all()
    load("../data/P_total.mat") % load training data
    load("../data/test_input.mat") % load test data
    load("../data/labels_binary_total.mat") % load labels for training data
    load("../data/target_filter.mat") % load Arial digits for filter
    % test_input=P;
    %test_lbls = test_labels(); % load labels for test data
    %test_lbls = ones(size(test_lbls)); % depends on the 
    %test_lbls_bin = binary_transform(test_lbls)';
    act_funct_strs = {'hardlim', 'purelin', 'logsig'}; % activation function strings
    % act_funct_strs = {'purelin', 'logsig'}; % activation function strings
    %% classifier 1: associative memory filter + classifier
    % assoc = associative_memory_filter(P_total, target_filter);
    % disp("after assoc");
    % output_filter = assoc * P_total;

    % net = binary_perceptron_filter(P_total, target_filter); % train filter with the desired Arial output
    % disp("after perceptron");
    % save("../data/perceptron.mat", "net");
    % load("../data/perceptron.mat");
    % output_filter_perceptron = sim(net, P_total);
    %! to be put in action again
    % for i = 1:length(act_funct_strs)
    %     act_funct_str = act_funct_strs{i};
    %     net = classifier_one_layer(output_filter, labels_binary_total, act_funct_strs{i}, true);
    %     save("../data/1layer_C_"+act_funct_str+"_no_softmax.mat", "net");
    %     disp("saved with the softmax");
    %     net = classifier_one_layer(output_filter, labels_binary_total, act_funct_strs{i}, false);
    %     save("../data/1layer_C_"+act_funct_str+"_softmax.mat", "net");
    % end
    %! do not mix softmax with hardlim function; it does not make sense

    %! to be put in action again
    % for i = 1:length(act_funct_strs)
    %     act_funct_str = act_funct_strs{i};
    %     net = classifier_one_layer(output_filter_perceptron, labels_binary_total, act_funct_strs{i}, true);
    %     after_train = sim(net, output_filter_perceptron);
    %     [c, cm, ind, per] = confusion(labels_binary_total, after_train);
    %     fprintf("Accuracy: %f\n", 1-c);
    %     save("../data/perceptron1C_"+act_funct_str+"_no_softmax.mat", "net");
    %     disp("saved with the softmax");
    %     net = classifier_one_layer(output_filter_perceptron, labels_binary_total, act_funct_strs{i}, false);
    %     after_train = sim(net, output_filter_perceptron);
    %     [c, cm, ind, per] = confusion(labels_binary_total, after_train);
    %     fprintf("Accuracy: %f\n", 1-c);
    %     save("../data/perceptron1C_"+act_funct_str+"_softmax.mat", "net");
    % end

    % classifier 2: no filtering; one layer
    % for i = 1:length(act_funct_strs)
    %     act_funct_str = act_funct_strs{i};
    %     net = classifier_one_layer(P_total, labels_binary_total, act_funct_str, false); %! working
    %     save("../data/1layer_"+act_funct_str+"_no_softmax.mat", "net");
    %     net = classifier_one_layer(P_total, labels_binary_total, act_funct_str, true); %! working
    %     save("../data/1layer_"+act_funct_str+"_softmax.mat", "net");
    % end
    % classifier 3: no filtering; two layers
    for i = 1:length(act_funct_strs)
        act_funct_str = act_funct_strs{i};
        net = classifier_two_layers(P_total, labels_binary_total, act_funct_str, false);
        after_train = sim(net, P_total);
        [c, cm, ind, per] = confusion(labels_binary_total, after_train);
        fprintf("Accuracy: %f\n", 1-c);
        save("../data/2layer_"+act_funct_str+"_no_softmax.mat", "net");
        net = classifier_two_layers(P_total, labels_binary_total, act_funct_str,true);
        after_train = sim(net, P_total);
        [c, cm, ind, per] = confusion(labels_binary_total, after_train);
        fprintf("Accuracy: %f\n", 1-c);
        save("../data/2layer_"+act_funct_str+"_softmax.mat", "net");
    end
    % % classifier 4: using patternnet
    % net = pattern_net(P_total, labels_binary_total);
    % save("../data/pattern.mat", "net");
end