function errors = train_all_fis(training_data, test_data)
    opt_methods = [0,1];
    op_method_names = ["Backpropagation", "Hybrid"];
    clustering_methods = ["GridPartition", "SubtractiveClustering", "FCMClustering"];
    train_input = training_data(:, 1:end-1);
    train_output = training_data(:, end);
    test_input = test_data(:, 1:end-1);
    test_output = test_data(:, end);
    str_file = "fis/fis_results.txt";
    fileSseID = fopen(str_file, "a");
    for i = 1:3
        clustering_method = clustering_methods(i);
        genOptions = genfisOptions(clustering_method);
        fis = genfis(train_input, train_output,genOptions);
        for j = 0:1
            anOptions = anfisOptions("InitialFis", fis, "ValidationData", test_data, "OptimizationMethod", j);
            trained_fis = anfis(training_data, anOptions);
            str_file = "fis/"+clustering_method+"_"+j;
            writeFIS(trained_fis, str_file);
            result = evalfis(trained_fis,test_input);
            sse = sum((result - test_output).^2);% /numel(result);
            no_rules = size(trained_fis.rules, 2);
            fprintf(fileSseID,"Clustering method: %s; opt method: %s, no. of rules:%d, sse: %g\n", clustering_method, op_method_names(j+1), no_rules, sse);
        end
    end
end