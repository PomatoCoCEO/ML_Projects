function register_fis_results(clustering_method, no_rules, type_function, opt_method, performance)
    % register: no of rules (3, 5 or 7)
    % scale factor 1: for output magnification
    % scale factor 2: for actuator magnification
    fileID = fopen('register_2.csv','a');
    fprintf(fileID,'%s,%d,%s,%s,%g\n',clustering_method,no_rules, type_function,opt_method,performance);
end