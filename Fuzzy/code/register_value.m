function register_values(no_inputs, scale1, scale2, performance)
    % register: no of rules (3, 5 or 7)
    % scale factor 1: for output magnification
    % scale factor 2: for actuator magnification
    fileID = fopen('register.csv','a');
    fprintf(fileID,'%d,%f,%f,%f\n',no_inputs,scale1, scale2, performance);
end