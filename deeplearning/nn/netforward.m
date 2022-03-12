function [net, outputs, As] = netforward(net, inputs, istraining)

if nargin < 3
    istraining = 0;
end

As = cstack;
As.empty();
activations = inputs;
for i = 1:numel(net.layers)
    layer = net.layers{i};
    [layer, activations] = layerforward(layer, activations);
    net.layers{i} = layer;
    if (istraining == 1)
        As.push(activations)
    end
end
outputs = activations;