function register_value(no_inputs, type_system, type_function, scale1, scale2, sim_time, performance)
    % register: no of rules (3, 5 or 7)
    % scale factor 1: for output magnification
    % scale factor 2: for actuator magnification
    fileID = fopen('register_2.csv','a');
    fprintf(fileID,'%d,%s,%s,%f,%f,%d,%f\n',no_inputs,type_system, type_function,scale1, scale2,sim_time,performance);
end