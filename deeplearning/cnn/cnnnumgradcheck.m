function cnnnumgradcheck(net,y,X)

    epsilon = 1e-4;
    er      = 1e-7;
    n = numel(net.layers);

    for l = n : -1 : 2
        if strcmp(net.layers{l}.type, 'dense')
            for j = 1 : numel(net.layers{l}.b)
                net_m = net; net_p = net;
                net_p.layers{l}.b(j) = net_m.layers{l}.b(j) + epsilon;
                net_m.layers{l}.b(j) = net_m.layers{l}.b(j) - epsilon;
                net_m = cnnff(net_m, X); net_m = cnnbp(net_m, y);
                net_p = cnnff(net_p, X); net_p = cnnbp(net_p, y);
                d = (net_p.loss - net_m.loss) / (2 * epsilon);
                e = abs(d - net.layers{l}.db(j));
                if e > er
                    error('numerical gradient checking failed');
                end
            end
            
            for i = 1 : size(net.layers{l}.W, 1)
                for u = 1 : size(net.layers{l}.W, 2)
                    net_m = net; net_p = net;
                    net_p.layers{l}.W(i, u) = net_m.layers{l}.W(i, u) + epsilon;
                    net_m.layers{l}.W(i, u) = net_m.layers{l}.W(i, u) - epsilon;
                    net_m = cnnff(net_m, X); net_m = cnnbp(net_m, y);
                    net_p = cnnff(net_p, X); net_p = cnnbp(net_p, y);
                    d = (net_p.loss - net_m.loss) / (2 * epsilon);
                    e = abs(d - net.layers{l}.dW(i, u));
                    if e > er
                        error('numerical gradient checking failed');
                    end
                end
            end
        elseif strcmp(net.layers{l}.type, 'conv')
            for j = 1 : numel(net.layers{l}.A)
                net_m = net; net_p = net;
                net_p.layers{l}.b{j} = net_m.layers{l}.b{j} + epsilon;
                net_m.layers{l}.b{j} = net_m.layers{l}.b{j} - epsilon;
                net_m = cnnff(net_m, X); net_m = cnnbp(net_m, y);
                net_p = cnnff(net_p, X); net_p = cnnbp(net_p, y);
                d = (net_p.loss - net_m.loss) / (2 * epsilon);
                e = abs(d - net.layers{l}.db{j});
                if e > er
                    error('numerical gradient checking failed');
                end
                for i = 1 : numel(net.layers{l - 1}.A)
                    for u = 1 : size(net.layers{l}.W{i}{j}, 1)
                        for v = 1 : size(net.layers{l}.W{i}{j}, 2)
                            net_m = net; net_p = net;
                            net_p.layers{l}.W{i}{j}(u, v) = net_p.layers{l}.W{i}{j}(u, v) + epsilon;
                            net_m.layers{l}.W{i}{j}(u, v) = net_m.layers{l}.W{i}{j}(u, v) - epsilon;
                            net_m = cnnff(net_m, X); net_m = cnnbp(net_m, y);
                            net_p = cnnff(net_p, X); net_p = cnnbp(net_p, y);
                            d = (net_p.loss - net_m.loss) / (2 * epsilon);
                            e = abs(d - net.layers{l}.dW{i}{j}(u, v));
                            if e > er
                                error('numerical gradient checking failed');
                            end
                        end
                    end
                end
            end
        elseif strcmp(net.layers{l}.type, 'pool')
           
        end
    end
%    keyboard
end
