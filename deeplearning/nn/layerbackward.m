function [layer, nextdelta] = layerbackward(layer, delta, A)

delta = delta .* fderiv(A, layer.activation);
layer.weights_grads = (1/layer.batchsize)*layer.X' * delta;
layer.biases_grads = mean(delta);
nextdelta = delta * layer.weights';