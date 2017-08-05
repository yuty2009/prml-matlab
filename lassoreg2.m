%% Least square regression with norm1 regularization
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% b: P by 1 regression coefficients
% b0: the intercept
%
% Lasso regularization can be re-formulated as
% g(x)  =  L(x)  +  \lambda \sqrt {x^T x + \epsilon}
% In such a case, the later item is differentiable and the optimzation
% problem can be solved by an iterated algorithm
function [varargout] = lassoreg2(y,X,lambda)

if nargin < 3, lambda = 1e-4; end

PHI = cat(2, ones(size(X,1),1), X); % add a constant column to cope with bias
[N, P] = size(PHI);

epsilon = 1e-3;
w = ones(P,1);

[cost,grad] = funcCost(w,y,PHI,epsilon,lambda);
grad1 = numgrad(@(p)funcCost(p,y,PHI,epsilon,lambda),w);
disp([grad grad1]); 
diff = norm(grad1-grad)/norm(grad1+grad);
disp(diff);

options.maxIter = 500;
options.display = 'on';
options.method = 'cg';
[w,cost] = minFunc(@(p)funcCost(p,y,PHI,epsilon,lambda), ...
    w,options);

b = w(2:P);
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

function [cost,grad] = funcCost(w,y,X,epsilon,lambda)

    gamma = w'*w + epsilon;
    grad = X'*(X*w-y) + lambda*w*gamma^(-1/2);
    cost = 0.5*(y-X*w)'*(y-X*w) + lambda*gamma^(1/2);
end
