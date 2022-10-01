function f=associative_memory_filter()
    load("target_filter.mat");
    load("weights_assoc.mat");
    load("P.mat");
    estimate = weights_assoc * P;
    f = estimate; 
    % returns the estimate associated to each digit that was drawn
    % the weights_assoc matrix was calculated with the pseudo_inverse method: W = T * pinv(P)
    % where P was the (256 * 500) input matrix and T was the (10 * 256) "perfect" matrix
end