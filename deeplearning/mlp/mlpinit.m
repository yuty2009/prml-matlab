%% Initialize a Multiple Layer Perceptron network
%  SN: the size of each layer (including input,hidden and output)
%  inittype: how to initialize the weights of each layer
function mlp = mlpinit(SN,inittype)

mlp.SN = SN;
mlp.NL = numel(SN); % No. of layers

mlp.TF = 'sigmoid'; % activation function of hidden layers
mlp.oTF = 'sigmoid'; % output function (linear,sigmoid,softmax)
mlp.lrate = 1; % learning rate for stochastic gradient descent
mlp.dropout = 0; % dropout level
mlp.corruption = 0; % corruption level for denoising AutoEncoders
mlp.lambda = 0; % L2 weight decay coefficient
mlp.beta = 0; % L1 non-sparsity penalty
mlp.sparsity = 0.05; % sparsity target;
mlp.momentum = 0.5; % momentum for RBMs
mlp.testing = 0;

if nargin < 2
    inittype = 'uniform';
end

switch(inittype)
    case 'zero'
        for L = 1:mlp.NL-1
            Sj = mlp.SN(L);
            Si = mlp.SN(L+1);
            mlp.W{L} = zeros(Sj,Si);
            mlp.b{L} = zeros(1,Si);
            mlp.dW{L} = zeros(Sj,Si);
            mlp.db{L} = zeros(1,Si);
        end
    case 'norm'
        epsilon = 1;
        for L = 1:mlp.NL-1
            Sj = mlp.SN(L);
            Si = mlp.SN(L+1);
            mlp.W{L} = epsilon*randn(Sj,Si);
            mlp.b{L} = zeros(1,Si);
            mlp.dW{L} = zeros(Sj,Si);
            mlp.db{L} = zeros(1,Si);
        end
    case 'uniform'
        r  = sqrt(6)/sqrt(mlp.SN(1)+mlp.SN(end)+1);
        for L = 1:mlp.NL-1
            Sj = mlp.SN(L);
            Si = mlp.SN(L+1);
            mlp.W{L} = rand(Sj,Si) * 2 * r - r;
            mlp.b{L} = zeros(1,Si);
            mlp.dW{L} = zeros(Sj,Si);
            mlp.db{L} = zeros(1,Si);
        end
end
