%% Train a single Restricted Boltzmann Machine
%      X: N by P input matrix, where N is the No. of samples
%          and P is the input dimension
%      rbm: the trained RBM model,
function rbm = rbmtrain(rbm,X,opts)

assert(isfloat(X), 'X must be float');
% assert(all(X(:)>=0) && all(X(:)<=1), 'all data in X must be in [0:1]');

N = size(X,1);
numbatches = N/opts.batchsize;
assert(rem(numbatches, 1) == 0, 'numbatches not integer');
for i = 1:opts.maxepochs
    perm = randperm(N);
    error = 0;
    for j = 1:numbatches
        batch = X(perm((j-1)*opts.batchsize+(1:opts.batchsize)), :);
        
        v1 = batch;
        h1 = sigmrnd(repmat(rbm.b,opts.batchsize,1) + v1*rbm.W);
        v2 = sigmrnd(repmat(rbm.a,opts.batchsize,1) + h1*rbm.W');
        h2 = sigmoid(repmat(rbm.b,opts.batchsize,1) + v2*rbm.W); % real value
        
        c1 = v1'*h1;
        c2 = v2'*h2;
        
        rbm.dW = rbm.momentum*rbm.dW + rbm.lrate*(c1 - c2)/opts.batchsize;
        rbm.da = rbm.momentum*rbm.da + rbm.lrate*sum(v1 - v2)/opts.batchsize;
        rbm.db = rbm.momentum*rbm.db + rbm.lrate*sum(h1 - h2)/opts.batchsize;
        
        rbm.W = rbm.W + rbm.dW;
        rbm.a = rbm.a + rbm.da;
        rbm.b = rbm.b + rbm.db;
        
        error = error +  sum(sum((v1 - v2) .^ 2))/opts.batchsize;
    end
    
    if isfield(opts,'verbose') && strcmpi(opts.verbose,'on')
        disp(['Epoch ' num2str(i) '/' num2str(opts.maxepochs) ...
            '. Average reconstruction error is: ' num2str(error/numbatches)]);
    end
end
