function f=associative_memory_filter(P,T)
    % load("target_filter.mat");
    % load("weights_assoc.mat");
    % load("P.mat");
    weights = T * pinv(P);
    % estimate = weights_assoc * P;

    f = weights; 
    % returns the estimate associated to each digit that was drawn
    % the weights_assoc matrix was calculated with the pseudo_inverse method: W = T * pinv(P)
    % where P was the (256 * 500) input matrix and T was the (10 * 256) "perfect" matrix
end