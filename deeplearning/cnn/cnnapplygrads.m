function cnn = cnnapplygrads(cnn,opts)

for L = 2:cnn.NL
    if strcmpi(cnn.layers{L}.type, 'dense') % fully-connected layer
        cnn.layers{L}.vW = cnn.momentum*cnn.layers{L}.vW + opts.lr * cnn.layers{L}.dW;
        cnn.layers{L}.W = cnn.layers{L}.W - cnn.layers{L}.vW;
        cnn.layers{L}.vb = cnn.momentum*cnn.layers{L}.vb + opts.lr * cnn.layers{L}.db;
        cnn.layers{L}.b = cnn.layers{L}.b - cnn.layers{L}.vb;
        
    elseif strcmpi(cnn.layers{L}.type, 'conv') % convolutional layer
        for j = 1:numel(cnn.layers{L}.A)
            for i = 1:numel(cnn.layers{L-1}.A)
                cnn.layers{L}.vW{i}{j} = cnn.momentum*cnn.layers{L}.vW{i}{j} + opts.lr * cnn.layers{L}.dW{i}{j};
                cnn.layers{L}.W{i}{j} = cnn.layers{L}.W{i}{j} - cnn.layers{L}.vW{i}{j};
            end
            cnn.layers{L}.vb{j} = cnn.momentum*cnn.layers{L}.vb{j} + opts.lr * cnn.layers{L}.db{j};
            cnn.layers{L}.b{j} = cnn.layers{L}.b{j} - cnn.layers{L}.vb{j};
        end
    end
end
