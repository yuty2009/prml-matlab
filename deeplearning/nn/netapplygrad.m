function net = netapplygrad(net, lr)

for i = 1:numel(net.layers)
    layer = net.layers{i};
    layer = layerapplygrad(layer, lr);
    net.layers{i} = layer;
end