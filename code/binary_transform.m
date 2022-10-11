function B = binary_transform(t)
    a = max(t);
    [s, ~] = size(t);
    B = zeros(s,a);
    for i = 1:s
        B(i,t(i)) = 1;
    end
end