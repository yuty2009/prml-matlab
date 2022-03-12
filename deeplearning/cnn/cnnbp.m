function cnn = cnnbp(cnn, y)

N = size(y,2); % number of training samples
ytrue = y;
ypred = cnn.layers{cnn.NL}.A{1};
assert(size(ypred,1) == size(ytrue,1) && size(ypred,2) == size(ytrue,2), ...
    'Shape disagreement for ypred and ytrue');

switch(cnn.losstype)
    case 'linear' % mean square loss
        cnn.layers{cnn.NL}.d{1} = ypred - ytrue; % output delta
        cnn.loss = 0.5*(1/N)*sum((ytrue(:)-ypred(:)).^2);
    case 'sigmoid' % mean square loss
        dfZ = ypred.*(1-ypred);
        cnn.layers{cnn.NL}.d{1} = (ypred - ytrue).*dfZ; % output delta
        cnn.loss = 0.5*(1/N)*sum((ytrue(:)-ypred(:)).^2);
    case 'softmax' % cross entropy loss
        cnn.layers{cnn.NL}.d{1} = ypred - ytrue; % output delta
        cnn.loss = -(1/N)*ytrue(:)'*log(ypred(:));
    case 'custom' % customized loss function
        % loss and delta are given outside this function
end

for L = cnn.NL:-1:1
    if strcmpi(cnn.layers{L}.type, 'dense') % fully-connected layer
        % numel(cnn.layers{L-1}.A) == 1 means a fully-connected layer
        % otherwise a pooling layer for layer L-1
        if (numel(cnn.layers{L-1}.A) == 1)
            cnn.layers{L-1}.fv = cnn.layers{L-1}.A{1};
        end
        cnn.layers{L}.dW = (1/N)*cnn.layers{L}.d{1}*(cnn.layers{L-1}.fv)';
        cnn.layers{L}.db = mean(cnn.layers{L}.d{1}, 2);
        
        % delta for fully-connected layer
        cnn.layers{L-1}.fvd = cnn.layers{L}.W'*cnn.layers{L}.d{1};
        if isfield(cnn.layers{L-1}, 'TF')
            cnn.layers{L-1}.fvd = cnn.layers{L-1}.fvd ...
                .* fderiv(cnn.layers{L-1}.fv, cnn.layers{L-1}.TF);
        end
        if isfield(cnn.layers{L-1},'dropout') && cnn.dropout > 0
            cnn.layers{L-1}.fvd = cnn.layers{L-1}.fvd  .* cnn.layers{L-1}.dropmask{1};
        end
        %  reshape feature vector deltas into output map style
        sa = size(cnn.layers{L-1}.A{1});
        fvnum = prod(sa(1:end-1));
        for j = 1:numel(cnn.layers{L-1}.A)
            cnn.layers{L-1}.d{j} = reshape(cnn.layers{L-1}.fvd(((j-1)*fvnum+1):j*fvnum, :), sa);
        end
        
    elseif strcmpi(cnn.layers{L}.type, 'conv') % convolutional layer
        for j = 1:numel(cnn.layers{L}.A)
            d = fderiv(cnn.layers{L}.A{j}, cnn.layers{L}.TF) .*  ...
                expand(cnn.layers{L+1}.d{j}, [cnn.layers{L+1}.kernelsize 1]);
            if ~isfield(cnn.layers{L+1},'method') || strcmpi(cnn.layers{L+1}.method, 'mean') % mean pooling
                cnn.layers{L}.d{j} = d / prod(cnn.layers{L+1}.kernelsize);
            elseif strcmpi(cnn.layers{L+1}.method, 'max') % max pooling
                cnn.layers{L}.d{j} = d .* cnn.layers{L+1}.posmatrix{j};
            elseif strcmpi(cnn.layers{L+1}.method, 'stochastic') % stochastic pooling
                cnn.layers{L}.d{j} = d .* cnn.layers{L+1}.posmatrix{j};
            end
        end
        
    elseif strcmpi(cnn.layers{L}.type, 'pool') % sub-sampling layer
        % deltas have already been calculated if the next layer is a
        % fully-connected layer
        if strcmpi(cnn.layers{L+1}.type, 'conv')
            if strcmpi(cnn.layers{L+1}.method, 'valid')
                bpmethod = 'full';
            elseif strcmpi(cnn.layers{L+1}.method, 'same')
                bpmethod = 'same';
            end
            for i = 1:numel(cnn.layers{L}.A)
                z = zeros(size(cnn.layers{L}.A{1}));
                for j = 1:numel(cnn.layers{L+1}.A)
                     z = z + convn(cnn.layers{L+1}.d{j}, flipall(cnn.layers{L+1}.W{i}{j}), bpmethod);
                end
                cnn.layers{L}.d{i} = z;
            end
        end
    end
end

for L = 2:cnn.NL
    if strcmpi(cnn.layers{L}.type, 'conv') % convolutional layer
        for j = 1:numel(cnn.layers{L}.A)
            for i = 1:numel(cnn.layers{L-1}.A)
                cnn.layers{L}.dW{i}{j} = convn(flipall(cnn.layers{L-1}.A{i}), cnn.layers{L}.d{j}, cnn.layers{L}.method)/N;
            end
            cnn.layers{L}.db{j} = sum(cnn.layers{L}.d{j}(:))/N;
        end
    end
end
