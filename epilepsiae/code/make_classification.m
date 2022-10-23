function binary_classif = make_classification(Trg)
    sz = size(Trg);
    curr_class = Trg(1); % first value
    num_class = 1;
    aid_array = [];
    tot_coisos = 0;
    for i = 2:sz(1)
        val = Trg(i);
        if curr_class == val
            num_class = num_class + 1;
        else
            aid_array = [aid_array; [curr_class num_class]];
            tot_coisos = tot_coisos + num_class;
            num_class= 1;
            curr_class = val;
        end
    end
    aid_array = [aid_array; [curr_class num_class]]
    sz_aid = size(aid_array);
    
    binary_classif = [];
    pre_ictal_val = 900;
    post_ictal_val = 300;
    for i = 1:sz_aid(1)
        curr = aid_array(i,1);
        num = aid_array(i,2);
        if curr == 0
            if i==1 && sz_aid(1) >=2 % start of the series, assume values alternate
                if num > pre_ictal_val
                    aid = zeros(4, num-pre_ictal_val);
                    aid(1,:) = 1; % interictal phase
                    binary_classif = [binary_classif, aid];
                end
                aid2 = zeros(4, min(pre_ictal_val, num));
                aid2(2,:)=1;
                binary_classif = [binary_classif, aid2]; % pre ictal phase
            elseif i > 1
                if sz_aid(1) >= i+1 && num > pre_ictal_val + post_ictal_val
                    aid1 = zeros(4, post_ictal_val);
                    aid1(4,:)=1; % post ictal
                    aid2 = zeros(4, num-pre_ictal_val-post_ictal_val);
                    aid2(1,:)=1; % inter-ictal
                    aid3 = zeros(4, pre_ictal_val);
                    aid3(2,:) =1; % pre ictal
                    binary_classif = [binary_classif, aid1, aid2, aid3];
                elseif num > post_ictal_val
                    aid1 = zeros(4, post_ictal_val);
                    aid1(4,:)=1; % post ictal
                    aid3 = zeros(4, num-post_ictal_val);
                    aid3(1,:)=1; % inter-ictal
                    binary_classif = [binary_classif, aid1,aid3];
                end
            end
        else 
            % curr = 1; add the corresponding 
            aid = zeros(4, num);
            aid(3,:) = 1;
            binary_classif = [binary_classif aid];
        end
    end
end