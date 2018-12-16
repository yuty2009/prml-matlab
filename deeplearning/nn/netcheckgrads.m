function netcheckgrads(net, inputs, outputs)

epsilon = 1e-4;
tolerence = 1e-8;
[net, act, as] = netforward(net, inputs, 1);
delta = lderiv(act, outputs, net.loss);
net = netbackward(net, delta, as);
for l = 1:numel(net.layers)
    for i = 1:size(net.layers{l}.weights, 1)
        for j = 1:size(net.layers{l}.weights, 2)
            net_p = net; net_m = net;
            net_p.layers{l}.weights(i,j) = net_p.layers{l}.weights(i,j) + epsilon;
            net_m.layers{l}.weights(i,j) = net_m.layers{l}.weights(i,j) - epsilon;
            [net_p, act_p, ~] = netforward(net_p, inputs, 1);
            [net_m, act_m, ~] = netforward(net_m, inputs, 1);
            loss_p = lvalue(act_p, outputs, net_p.loss);
            loss_m = lvalue(act_m, outputs, net_m.loss);
            dWij = (loss_p - loss_m)/(2*epsilon);
            e = abs(dWij - net.layers{l}.weights_grads(i,j));
            if (e > tolerence)
                disp('numerical gradient checking failed')
            end
        end
    end
end