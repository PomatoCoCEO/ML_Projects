function C = my_classify(classifier_name, act_func)
    disp(act_func)
    if classifier_name == "AssMem"
        disp(1)

    elseif classifier_name == "Percept"
         disp(2)
    elseif classifier_name == "OneLayer"
        disp(3)
    elseif classifier_name == "TwoLayer"
        disp(4)
    elseif classifier_name == "Pattern"
        disp(5)
    end
end