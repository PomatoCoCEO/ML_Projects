function C = my_classify(classifier_name, act_func, with_softmax)
    folderName = "nets_1000ep_1000in";
    load("../data/P.mat");
    test_input=P;
    load("../data/ind.mat");
    load("../data/PerfectArial.mat");

    if classifier_name == "AssMem"
        load("../data/"+folderName+"/AFW.mat"); % associative layer
        net_name = "1layer_c_";
        net_name = net_name + act_func;
        if with_softmax
            net_name = net_name+ "_softmax.mat";
        else
            net_name = net_name+ "_no_softmax.mat";
        end
        load("../data/"+ folderName+"/"+net_name);
        output_assoc = weights * test_input;
        out_sim = sim(net, output_assoc);

    elseif classifier_name == "Percept"
        load("../data/"+folderName+"/perceptron.mat"); 
        perceptron = net;
        net_name = "perceptron1C_";
        net_name = net_name + act_func;
        if with_softmax
            net_name = net_name+ "_softmax.mat";
        else
            net_name = net_name+ "_no_softmax.mat";
        end
        load("../data/"+ folderName+"/"+net_name);
        output_bp = sim(perceptron, test_input);
        out_sim = sim(net, output_bp);
    elseif classifier_name == "OneLayer"
        net_name = "1layer_";
        net_name = net_name + act_func;
        if with_softmax
            net_name = net_name+ "_softmax.mat";
        else
            net_name = net_name+ "_no_softmax.mat";
        end
        load("../data/"+ folderName+"/"+net_name);
        out_sim = sim(net, test_input);
        
        

    elseif classifier_name == "TwoLayer"
        net_name = "2layer_";
        net_name = net_name + act_func;
        if with_softmax
            net_name = net_name+ "_softmax.mat";
        else
            net_name = net_name+ "_no_softmax.mat";
        end
        load("../data/"+ folderName+"/"+net_name);
        out_sim = sim(net, test_input);
        
    elseif classifier_name == "Pattern"
        net_name = "pattern.mat";
        load("../data/"+ folderName+"/"+net_name);
        out_sim = sim(net, test_input);
    end


    

    conv_out =  convert_output(out_sim);
    digits_out = one_hot_to_digit(conv_out);
    
    predicted = zeros(size(P));
    sp = size(P);
    for i= 1:sp(2)
        predicted(:,i)= Perfect(:,digits_out(i));
    end
    predicted=predicted(:,ind);
    figure;
    showim(predicted);
end