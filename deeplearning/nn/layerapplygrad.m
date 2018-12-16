function layer = layerapplygrad(layer, lr, wd)

if ~exist('lr', 'var')
    lr = 1e-4;
end
if ~exist('wd', 'var')
    wd = 4e-4;
end

layer.weights = layer.weights*(1. - wd);
layer.biases = layer.biases*(1. - wd);
layer.weights = layer.weights - lr * layer.weights_grads;
layer.biases = layer.biases - lr * layer.biases_grads;
layer.weights_grads = zeros(size(layer.weights_grads));
layer.biases_grads = zeros(size(layer.biases_grads));