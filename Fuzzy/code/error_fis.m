function err = error_fis(fis_trained, test_data)
    result = evalfis(fis_trained, test_data(:,1:end-1));
    err = sum((test_data(:,end) - result).^2);
end