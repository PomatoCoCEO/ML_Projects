function target =  perfect_target(s, labels)
    load('../data/PerfectArial.mat')
    target=[];
    for r=1:s
        d = labels(r);
        if d==0
            target=[target, Perfect(:,10)];
        else 
            target=[target, Perfect(:,d)];
        end
    end
end