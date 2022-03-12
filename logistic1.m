%% Logistic regression for binary classification (Page 205-208 of PRML)
% Iterative reweighted least square (IRLS) by Newton-Raphson
% iterative optimization scheme.
% w_new = w_old - (PHI'*R*PHI)^(-1)*PHI'*(y-t);
% 
% X: N by P design matrix with N samples of M features
% t: N by 1 target values {0,1}
% b: P by 1 weight vector
% b0: bias
function [varargout] = logistic1(t,X,lambda)

if nargin < 3, lambda = 1e-4; end

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N,P] = size(PHI);
% t(find(t==-1)) = 0; % the class label should be [1 0]

% initialization
w = zeros(P,1);

[cost,grad] = funcCost(w,lambda,PHI,t);
grad1 = numgrad(@(p)funcCost(p,lambda,PHI,t),w);
disp([grad grad1]); 
diff = norm(grad1-grad)/norm(grad1+grad);
disp(diff); 

options.maxIter = 200;
options.display = 'on';

[w,cost] = minFunc(@(p)funcCost(p,lambda,PHI,t),w,options);

b = w(2:end);
b0 = w(1);

if nargout == 1
    model.b = b;
    model.b0 = b0;
    varargout{1} = model;
elseif nargout == 2
    varargout{1} = b;
    varargout{2} = b0;
end

end

%% min sum(log(1+exp(-t.*(PHI*W)))) + lambda*norm(w); 
function [cost,grad] = funcCost(w,lambda,PHI,t)
    
    z = t.*(PHI*w);
    y = 1./(1+exp(-z));
    grad = -PHI'*(t.*(1-y)) + lambda*w;
    
    cost = -sum(log(y)) + 0.5*lambda*(w'*w);
 
end
