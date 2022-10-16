function test_all(folderName)
    networkFolderPath = "../data/"+folderName;
    files=ls("../data/"+folderName+"/*.mat");
    load("../data/P.mat"); % load test data
    test_input=P;
    test_lbls = test_labels(); % load labels for test data
    test_lbls_bin = binary_transform(test_lbls)';
    load("../data/"+folderName+"/AFW.mat"); % associative layer
    load("../data/"+folderName+"/perceptron.mat"); % associative layer
    perceptron = net;
    sf = size(files)
    for i = 1:sf(1)
        file=char(strtrim(files(i,:))); % falta fazer loading e avaliar a accuracy e etc,
        load("../data/"+folderName+"/"+file);
        c=1; cm=1; ind=1; per=1; real_output = [];
        if strcmp(file, "AFW.mat")
            continue
        end
        % tendo em conta os nomes
        if strcmp(file(1:9),'1layer_C_')
            output_assoc = weights * test_input;
            real_output = sim(net, output_assoc);
        elseif strcmp(file(1:10),'perceptron')
            % associative layer and perceptron filter
            output_assoc = sim(perceptron, test_input);
            real_output = sim(net, output_assoc);
        elseif strcmp(file(1:7),"1layer_") || strcmp(file(1:8), "2layers_")  
            % 1 and layer classifier
            real_output = sim(net, test_input);
        end
        fprintf("Classifier from %s after filter:\n" , net.name);
        fprintf("Accuracy: %f\n", (1-c));
        fprintf("Confusion matrix: \n");
        disp(cm);
        fprintf("\n");
    end
end
