%% Predict new samples with a trained mulit-layer perceptron 
%     (feed-forward) neural network model
%     mlp: the trained feed-forward neural network model
%     X: N by P input matrix, where N is the No. of samples
%     and P is the input dimension
%     y: N by 1 output matrix, where 1 is the output dimension
function varargout = mlppredict(mlp,X)

mlp.testing = 1;
mlp = mlpff(mlp,X);
mlp.testing = 0;

switch(mlp.oTF)
    case {'sigmoid','linear'}
        y = mlp.A{mlp.NL};
    case 'softmax'
        [dummy,y] = max(mlp.A{mlp.NL},[],2);
end

varargout{1} = y;
if nargout >= 2
    varargout{2} = mlp.A{mlp.NL};
end