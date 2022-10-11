function d = one_hot_to_digit(v)
    %assumes that v's size is 10 x number of outputs
    d = mod(find(v==1),10);
    d(d==0) = 10;
end