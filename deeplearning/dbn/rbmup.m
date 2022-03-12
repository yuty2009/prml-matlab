function Y = rbmup(rbm, X)
    Y = sigmoid(repmat(rbm.b, size(X, 1), 1) + X*rbm.W);
end