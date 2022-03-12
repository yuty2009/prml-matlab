function Y = rbmdown(rbm, X)
    Y = sigmoid(repmat(rbm.a, size(X, 1), 1) + X*rbm.W');
end
