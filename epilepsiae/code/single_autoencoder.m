function [encoder, encoded] = single_autoencoder(X, size)

% X isnt transposed
% but encoding shoulb use a transposed X
encoder = trainAutoencoder(X',size);
encoded = encode(encoder, X');

end