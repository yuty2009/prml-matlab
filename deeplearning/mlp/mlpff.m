%% Feed forward of the MLP
function mlp = mlpff(mlp,X)

N = size(X,1);

% allow different weight decay coefficients for different layers
if numel(mlp.lambda) == 1
    mlp.lambda = mlp.lambda*ones(mlp.NL-1,1);
end

% input corruption for denoising AE
if mlp.corruption > 0
    mlp.A{1} = X.*(rand(size(X))>mlp.corruption);
else
    mlp.A{1} = X; % aj: N by P input
end

for L = 2:mlp.NL-1
    Z = mlp.A{L-1}*mlp.W{L-1} + repmat(mlp.b{L-1},N,1); % N by Si
    mlp.A{L} = fvalue(Z,mlp.TF); % N by Si
    % dropout
    if mlp.dropout > 0
        if(mlp.testing)
            mlp.A{L} = mlp.A{L} .* (1 - mlp.dropout);
        else
            rng(0); % necessary for gradient checking
            mlp.dropmask{L-1} = rand(size(mlp.A{L})) > mlp.dropout;
            mlp.A{L} = mlp.A{L} .* mlp.dropmask{L-1};
        end
    end
end
Z = mlp.A{mlp.NL-1}*mlp.W{mlp.NL-1} + repmat(mlp.b{mlp.NL-1},N,1);
mlp.A{mlp.NL} = fvalue(Z,mlp.oTF);
