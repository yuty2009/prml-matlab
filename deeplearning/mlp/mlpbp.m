%% Error back propagation of MLP
function mlp = mlpbp(mlp,y)

N = size(y,1);

% error back propagation
switch(mlp.oTF)
    case 'linear'
        mlp.d{mlp.NL-1} = (mlp.A{mlp.NL} - y);
    case 'sigmoid'
        dfZ = mlp.A{mlp.NL}.*(1-mlp.A{mlp.NL});
        mlp.d{mlp.NL-1} = (mlp.A{mlp.NL} - y).*dfZ;
    case 'softmax'
        mlp.d{mlp.NL-1} = (mlp.A{mlp.NL} - y);
end

for L = mlp.NL-2:-1:1
    dfZ = fderiv(mlp.A{L+1},mlp.TF); % N by K
    if mlp.beta > 0 % sparsity coefficient
        rhos = mean(mlp.A{L+1}); % average activation of hidden layer
        sparseterm = -mlp.sparsity./rhos+(1-mlp.sparsity)./(1-rhos);
        mlp.d{L} = (mlp.d{L+1}*mlp.W{L+1}'+mlp.beta*repmat(sparseterm,N,1)).*dfZ;
    else
        mlp.d{L} = (mlp.d{L+1}*mlp.W{L+1}').*dfZ; % N by Si
    end
    
    if mlp.dropout > 0
        mlp.d{L} = mlp.d{L} .* mlp.dropmask{L};
    end
end
    
for L = 1:mlp.NL-1
    mlp.dW{L} = (1/N)*mlp.A{L}'*mlp.d{L} + mlp.lambda(L)*mlp.W{L}; % Sj by Si
    mlp.db{L} = mean(mlp.d{L}); % 1 by Si
end

weightdecay = 0;
for L = 1:mlp.NL-1
    weightdecay = weightdecay + 0.5*mlp.lambda(L)*sum(mlp.W{L}(:).^2);
end

sparsepenalty = 0;
if mlp.beta > 0
    for L = 1:mlp.NL-2
        rhos = mean(mlp.A{L+1}); % average activation of hidden layer
        sparsepenalty = sparsepenalty + mlp.beta*sum(mlp.sparsity.*log(mlp.sparsity./rhos) ...
             +(1-mlp.sparsity).*log((1-mlp.sparsity)./(1-rhos)));
    end
end

switch (mlp.oTF)
    case 'linear'
        mlp.cost = 0.5*(1/N)*sum((y(:)-mlp.A{mlp.NL}(:)).^2) ...
            + weightdecay + sparsepenalty;
    case 'sigmoid'
        mlp.cost = 0.5*(1/N)*sum((y(:)-mlp.A{mlp.NL}(:)).^2) ...
            + weightdecay + sparsepenalty;
    case 'softmax'
        mlp.cost = -(1/N)*y(:)'*log(mlp.A{mlp.NL}(:)) ...
            + weightdecay + sparsepenalty;
end
