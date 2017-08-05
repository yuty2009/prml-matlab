function cnn = cnnff(cnn,X)

inputmaps = 1;
inshape = size(X); % input dimensions
N = inshape(end);  % number of training samples
mapsize = inshape(1:end-1); % input dimension >= 3 is allowed

cnn.layers{1}.A{1} = X;
for L = 2:cnn.NL
    if strcmpi(cnn.layers{L}.type, 'dense') % fully-connected layer
        mapsize = cnn.layers{L}.outputsize;
        %  concatenate all end layer feature maps into vector
        cnn.layers{L-1}.fv = [];
        for j = 1:numel(cnn.layers{L-1}.A)
            sa = size(cnn.layers{L-1}.A{j});
            cnn.layers{L-1}.fv = [cnn.layers{L-1}.fv; ...
                reshape(cnn.layers{L-1}.A{j}, prod(sa(1:end-1)), sa(end))];
        end
        Z = cnn.layers{L}.W*cnn.layers{L-1}.fv + repmat(cnn.layers{L}.b,1,size(cnn.layers{L-1}.fv,2));
        cnn.layers{L}.A{1} = fvalue(Z', cnn.layers{L}.TF)';
        inputmaps = 1;
        
        if isfield(cnn.layers{L},'dropout') && cnn.dropout > 0
            % rng(0); % necessary for gradient checking
            cnn.layers{L}.dropmask{1} = rand(size(cnn.layers{L}.A{1})) > cnn.dropout;
            cnn.layers{L}.A{1} = cnn.layers{L}.A{1} .* cnn.layers{L}.dropmask{1};
        end
        
    elseif strcmpi(cnn.layers{L}.type, 'conv') % convolutional layer
        if strcmpi(cnn.layers{L}.method, 'valid')
            mapsize = mapsize - cnn.layers{L}.kernelsize + 1;
        end
        for j = 1:cnn.layers{L}.outputmaps
            z = zeros([mapsize N]);
            for i = 1:inputmaps
                z = z + convn(cnn.layers{L-1}.A{i},cnn.layers{L}.W{i}{j},cnn.layers{L}.method);
            end
            Z = z + cnn.layers{L}.b{j};
            cnn.layers{L}.A{j} = fvalue(Z, cnn.layers{L}.TF);
        end
        inputmaps = cnn.layers{L}.outputmaps;
        
    elseif strcmpi(cnn.layers{L}.type, 'pool') % sub-sampling layer
        mapsize = mapsize./cnn.layers{L}.kernelsize;
        for j = 1:inputmaps
            if ~isfield(cnn.layers{L},'method') || strcmpi(cnn.layers{L}.method, 'mean') % mean pooling
                z = averagepooling(cnn.layers{L-1}.A{j}, cnn.layers{L}.kernelsize);
            elseif strcmpi(cnn.layers{L}.method, 'max') % max pooling
                [z, maxpos] = maxpooling(cnn.layers{L-1}.A{j}, cnn.layers{L}.kernelsize);
                maxpos = sparse(ones(length(maxpos),1),maxpos,ones(length(maxpos),1),1,numel(cnn.layers{L-1}.A{j}));
                cnn.layers{L}.posmatrix{j} = reshape(full(maxpos),size(cnn.layers{L-1}.A{j}));
            elseif strcmpi(cnn.layers{L}.method, 'stochastic') % max pooling
                [z, randompos] = stochasticpooling(cnn.layers{L-1}.A{j}, cnn.layers{L}.kernelsize);
                randompos = sparse(ones(length(randompos),1),randompos,ones(length(randompos),1),1,numel(cnn.layers{L-1}.A{j}));
                cnn.layers{L}.posmatrix{j} = reshape(full(randompos),size(cnn.layers{L-1}.A{j}));
            end
            
            cnn.layers{L}.A{j} = z;
            
        end
    end
end

cnn.output = cnn.layers{cnn.NL}.A{1};
cnn.losstype = cnn.layers{cnn.NL}.TF;
