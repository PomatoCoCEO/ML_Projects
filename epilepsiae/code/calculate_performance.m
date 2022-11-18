function calculate_performance(conf_mat)


for i= 1:4 

    tp = conf_mat(i,i);
    fn = sum(conf_mat(:,i)) - tp;
    fp = sum(conf_mat(i,:)) - tp;
    tn = sum(sum(conf_mat)) - tp - fn -fp;
    

    sens = tp/(tp+fn);
    spec = tn/(tn+fp);
    
    
    printf("Sensitivity: %d. Specificity: %d", sens, spec);

end
end