function net = netbackward(net, delta, As)

for i = numel(net.layers):-1:1
    layer = net.layers{i};
    Al = As.pop();
    [layer, delta] = layerbackward(layer, delta, Al);
    net.layers{i} = layer;
end