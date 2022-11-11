function ew = get_error_weights(categorical_target)
    [GC, GR] = groupcounts(categorical_target)
    tot = size(GC,1);
    ew = zeros(tot,1);
    soma = sum(1/GC)
    for i = 1 : tot
        ew(GR(i)) = (1 / (GC(GR(i)))) / soma;
    end
end