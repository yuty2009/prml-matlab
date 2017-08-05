%% Train a multi-layer perceptron (feed-forward) neural network model
%     with the backpropagation algorithm
%     X: N by P input matrix, where N is the No. of samples
%        and P is the input dimension
%     y: N by K output matrix, where K is the output dimension
%     mlp: the trained feed-forward neural network model
function mlp = mlptrain(mlp,y,X,opts)

assert(isfloat(X), 'X must be float');

K = size(y,2);
% multi-class problem
if strcmpi(mlp.oTF,'softmax')
    if (K == 1)
        % K = length(unique(y));
        y = full(sparse(1:length(y),y,1));
    end
end

if ~isfield(opts,'method')
    opts.method = 'batch';
end

if ~isfield(opts,'stopeps')
    opts.stopeps = 1e-6;
end

% vetorize the params
mlp.theta = mlpparam(mlp,1,mlp.W,mlp.b);
mlp.vtheta = mlpparam(mlp,1,mlp.dW,mlp.db);

% training using mini-batch stochastic gradient descent
if strcmpi(opts.method, 'minibatch')
    N = size(X,1);
    numbatches = N/opts.batchsize;
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');

    opttheta = mlp.theta;
    for i = 1:opts.maxepochs
        tic;
        error = 0;
        perm = randperm(N);
        for j = 1:numbatches
            idx = perm((j-1)*opts.batchsize+(1:opts.batchsize));
            batch_X = X(idx, :);
            batch_y = y(idx, :);
            
            [cost,grad] = mlpcost(opttheta,mlp,batch_y,batch_X);
            
            mlp.vtheta = mlp.momentum*mlp.vtheta + mlp.lrate*grad;
            opttheta = opttheta - mlp.vtheta;
            
            error = error + cost;
        end
        toc;
        if strcmpi(opts.verbose,'on')
            disp(['Epoch ' num2str(i) '/' num2str(opts.maxepochs) ...
                '. Average cost is: ' num2str(error/numbatches)]);
        end
    end
        
% training using batch gradient descent
elseif strcmpi(opts.method, 'batch') 
    % Gradient checking for debug
    % [cost,grad] = mlpcost(mlp.theta,mlp,y,X);
    % grad1 = numgrad(@(p)mlpcost(p,mlp,y,X),mlp.theta);
    % disp([grad grad1]); 
    % diff = norm(grad1-grad)/norm(grad1+grad);
    % disp(diff); 

    options.maxIter = opts.maxepochs;
    options.display = opts.verbose;
    options.Method = opts.optmethod;
    [opttheta,cost] = minFunc(@(p)mlpcost(p,mlp,y,X),mlp.theta,options);
    
end

[mlp.W,mlp.b] = mlpparam(mlp,2,opttheta);

end

function [cost,grad] = mlpcost(theta,mlp,y,X)

    [mlp.W,mlp.b] = mlpparam(mlp,2,theta);
    
    mlp = mlpff(mlp,X);
    mlp = mlpbp(mlp,y);
    
    cost = mlp.cost;
    grad = mlpparam(mlp,1,mlp.dW,mlp.db);
end

