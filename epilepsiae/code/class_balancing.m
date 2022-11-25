function [x_sub, t_sub, t_sub_ohe ] = class_balancing(x, t,t_ohe, major_ratio)

idx = find(double(t)>1);
idx_1 =  find(double(t)==1);

n = size(idx_1,1);
k = floor(size(idx,1)/ (1 - major_ratio));

rp = randperm(n,floor(k* major_ratio));
selected_ones = idx_1(rp,:);
size(selected_ones);

size(idx);
new_idx = [selected_ones; idx];
new_idx = sort(new_idx);

x_sub= x(new_idx,:);
t_sub = t(new_idx,:);
t_sub_ohe = t_ohe(new_idx,:);


end