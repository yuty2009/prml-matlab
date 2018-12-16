function [layer, outputs] = layerforward(layer, inputs)

layer.X = reshape(inputs, layer.batchsize, []);
Z = layer.X * layer.weights + repmat(layer.biases, layer.batchsize,1);
outputs = fvalue(Z, layer.activation);
