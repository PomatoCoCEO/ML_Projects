function [sensitivity, specificity] =report_spec_sens_ohe(y_true, y_predicted, class_pos)
%REPORT_SPEC_SENS Prints the report on the specificity and sensitivity of a
%network
    [~,~, ~, per] = confusion(y_true, y_predicted);
    true_positives = per(class_pos, 3);
    true_negatives = per(class_pos, 4);
    false_positives = per(class_pos, 2);
    false_negatives = per(class_pos, 1);
    % no semicolon because printing is made easier
    sensitivity = (true_positives)/(true_positives + false_negatives);
    specificity = true_negatives/(true_negatives + false_positives);
    
end