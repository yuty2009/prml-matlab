%% Derivative of a specific transfer function on the given input
function Y = fderiv(Y1,func)
% Y1 = fvalue(X,func);
switch(func)
    case 'linear'
        Y = ones(size(Y1));
    case 'sigmoid'
        Y = Y1.*(1-Y1);
    case 'tanh'
        Y = 1 - Y1.^2;
    case 'tanh_opt'
        Y = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * Y1.^2);
    case 'ReLU'
        Y = Y1 > 0;
    case 'softmax'
        Y = ones(size(Y1));
    otherwise
        disp('unknown transfer function');
end
