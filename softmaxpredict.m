%% Softmax regression prediction
% X: N by P feature matrix, N number of samples, P number of features
% W: P by K regression coefficients
% y: N by 1 target vector in {1,...,K}
% py: N by K target probabilities, py(n,k) = p(y=k|x(n),params)
function [varargout] = softmaxpredict(X,W)

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N,P] = size(PHI);

Y = PHI*W;
Ytemp = bsxfun(@minus,Y,max(Y,[],2));
Ytemp = exp(Ytemp);
py = bsxfun(@rdivide,Ytemp,sum(Ytemp,2));

[dummy,y] = max(py,[],2);
        
varargout{1} = y;
if nargout > 1
    varargout{2} = py;
end