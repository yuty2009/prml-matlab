%% Softmax regression using gradient descend update
% X: N by P feature matrix, N number of samples, P number of features
% t: N by 1 class labels (t=k indicate belong to class k)
% lambda: regularization coefficient
% W: P by K regression coefficients
function [W] = softmax(t,X,lambda)

if nargin <= 2
    lambda = 1e-4;
end

K = length(unique(t));
T = full(sparse(1:length(t),t,1));
% add a constant column to cope with bias
PHI = cat(2,ones(size(X,1),1),X);
[N,P] = size(PHI);

% initialization
W = ones(P,K);

theta = W(:);

% [cost,grad] = funcCost(theta,lambda,PHI,T);
% grad1 = numgrad(@(p)funcCost(p,lambda,PHI,T),theta);
% disp([grad grad1]); 
% diff = norm(grad1-grad)/norm(grad1+grad);
% disp(diff); 

options.maxIter = 200;
options.display = 'on';

[opttheta,cost] = minFunc(@(p)funcCost(p,lambda,PHI,T), ...
    theta,options);

W = reshape(opttheta,P,K);

end
    
function [cost,grad] = funcCost(theta,lambda,PHI,T)
    
    [N,P] = size(PHI);
    K = size(T,2);
    
    W = reshape(theta,P,K);

    Z = PHI*W;
    Z = bsxfun(@minus, Z, max(Z,[],2));
    Y = exp(Z);
    Y = bsxfun(@rdivide, Y, sum(Y,2));

    dW = (1/N)*PHI'*(Y-T) + lambda*W;
    
    cost = -(1/N)*T(:)'*log(Y(:)) ...
     + 0.5*lambda*sum(W(:).^2);
 
    grad = dW(:);
end
