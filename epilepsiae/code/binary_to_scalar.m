function arr = binary_to_scalar(a)
    sz = size(a);
    arr = zeros(1,sz(1));
    for i = 1 : sz(1)
        aid = a(i,:);
        pos = find(aid == 1);
        arr(i)=pos(1);
    end
end