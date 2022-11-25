function acc = accuracy(y, target)
    %expects  y and target to be with size Noutputs x 1
    n = size(y,1);
    correct = sum(y==target);
    acc = correct/n;
end