function c = array_to_cell(a)
    % 
    c = {};
    sz = size(a);
    for i = 1:sz(1)
        c{i} = a(i,:)';
    end
end