function cnn = cnninit(cnn, outshape, inshape)

cnn.testing = 0;
cnn.dropout = 0;
cnn.momentum = 0;
inputmaps = 1;
mapsize = inshape(1:end-1); % input dimension >= 3 is allowed
assert(inshape(end)==outshape(end), ...
    ['The number of samples and labels should be the same but actual' ...
    'samples: ' num2str(inshape(1)) ' vs labels: ' num2str(outshape(1))]);

cnn.NL = numel(cnn.layers);
for L = 2:cnn.NL
    if strcmpi(cnn.layers{L}.type, 'dense') % fully-connected layer
        if ~isfield(cnn.layers{L},'TF')
            cnn.layers{L}.TF = 'sigmoid';
        end
        cnn.layers{L}.outputmaps = 1;
        
        insize = prod(mapsize)*inputmaps;
        outsize = cnn.layers{L}.outputsize;
        cnn.layers{L}.W = varinit([outsize, insize], 'normal_truncated', struct('std', 0.1));
        cnn.layers{L}.vW = varinit([outsize, insize], 'constant', struct('value', 0));
        % for ReLU, b must be non-zero intilized, ReLU is not differentiable at zero
        cnn.layers{L}.b = varinit([outsize,1], 'constant', struct('value', 0.1));
        cnn.layers{L}.vb = varinit([outsize,1], 'constant', struct('value', 0));
        mapsize = outsize;
        inputmaps = cnn.layers{L}.outputmaps;

    elseif strcmpi(cnn.layers{L}.type, 'conv') % convolutional layer
        if ~isfield(cnn.layers{L},'TF')
            cnn.layers{L}.TF = 'sigmoid';
        end
        if ~isfield(cnn.layers{L},'method')
            cnn.layers{L}.method = 'valid';
        end
        if length(cnn.layers{L}.kernelsize) == 1 % multi-dimensional non-square kernel is allowed
            cnn.layers{L}.kernelsize = cnn.layers{L}.kernelsize*ones(1,length(mapsize));
        end
        for j = 1:cnn.layers{L}.outputmaps
            for i = 1:inputmaps
                cnn.layers{L}.W{i}{j} = varinit(cnn.layers{L}.kernelsize, 'normal_truncated', struct('std', 0.1));
                cnn.layers{L}.vW{i}{j} = varinit(cnn.layers{L}.kernelsize, 'constant', struct('value', 0));
            end
            cnn.layers{L}.b{j} = varinit(1, 'constant', struct('value', 0.1));
            cnn.layers{L}.vb{j} = 0;
        end
        if strcmpi(cnn.layers{L}.method, 'valid')
            mapsize = mapsize - cnn.layers{L}.kernelsize + 1;
        end
        inputmaps = cnn.layers{L}.outputmaps;
        
    elseif strcmpi(cnn.layers{L}.type, 'pool') % pooling layer
        if ~isfield(cnn.layers{L},'method')
            cnn.layers{L}.method = 'mean';
        end
        if length(cnn.layers{L}.kernelsize) == 1 % different kernel parameters are allowed
            cnn.layers{L}.kernelsize = cnn.layers{L}.kernelsize*ones(1,length(mapsize));
        end
        mapsize = mapsize./cnn.layers{L}.kernelsize;
        assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(L) ' size must be integer while actual: ' num2str(mapsize)]);
    end
end

