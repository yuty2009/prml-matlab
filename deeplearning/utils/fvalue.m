%% Apply a specified transfer function on the given input
function Y = fvalue(X,func)

switch(func)
    case 'linear'
        Y = X;
    case 'sigmoid'
        Y = 1./(1+exp(-X));
    case 'tanh'
        Y = tanh(X);
    case 'tanh_opt'
        Y = 1.7159*tanh(2/3.*X);
    case 'ReLU'
        Y = max(X,0);
    case 'softmax'
        Y = bsxfun(@minus,X,max(X,[],2));
        Y = exp(Y);
        Y = bsxfun(@rdivide,Y,sum(Y,2));
    otherwise
        disp('unknown transfer function');
end
