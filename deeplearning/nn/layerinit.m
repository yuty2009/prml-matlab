function layer = layerinit(inshape, wshape, bshape, activation, dropout)

layer.inshape = inshape;
layer.batchsize = inshape(1);
if exist('activation', 'var')
    layer.activation = activation;
else
    layer.activation = 'linear';
end
if exist('dropout', 'var')
    layer.dropout = dropout;
else
    layer.dropout = 1;
end
layer.weights = weight_init(wshape);
layer.biases = bias_init(bshape);

layer.outshape = [layer.batchsize, wshape(2)];
layer.weights_grads = zeros(wshape);
layer.biases_grads = zeros(bshape);

end

function weights = weight_init(weight_shape)
    weights = 0.1*randn(weight_shape);
end

function biases = bias_init(bias_shape)
    biases = 0.1*randn(bias_shape);
end