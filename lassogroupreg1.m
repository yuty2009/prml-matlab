%% Least square regression with grouped lasso regularization
% The optimization objective is as following:
% w = {arg min}_w (||Xw-y||_2^2 + \lambda \sum_g^G ||w_{I_g}||_2)
% This can be re-formulated as:
% w = {arg min}_w (||Xw-y||_2^2 + \lambda \sum_g^G \sqrt{w_{I_g}^T w_{I_g}+\epsilon})
% Thus, in this case, the later item becomes differentiable now, which
% makes the problem can be solved by an iterated algorithm
% 
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% b: P by 1 regression coefficients
% b0: the intercept
function [varargout] = lassogroupreg1(y,X,group,lambda)

if nargin < 4, lambda = 1e-4; end

PHI = cat(2, ones(size(X,1),1), X); % add a constant column to cope with bias
[N, P] = size(PHI);

group = [0;group(:)]; % account for bias

epsilon = 1e-6;
w = ones(P,1);

[cost,grad] = funcCost(w,y,PHI,group,epsilon,lambda);
grad1 = numgrad(@(p)funcCost(p,y,PHI,group,epsilon,lambda),w);
disp([grad grad1]); 
diff = norm(grad1-grad)/norm(grad1+grad);
disp(diff);

options.maxIter = 200;
options.display = 'on';
[w,cost] = minFunc(@(p)funcCost(p,y,PHI,group,epsilon,lambda), ...
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

function [cost,grad] = funcCost(w, y, X, group, epsilon, lambda)
    groupid = unique(group);
    NG1 = length(groupid);

    sum_w = 0;
    grad = zeros(length(w),1);
    for i = 1:NG1
        index_ig = find(group == groupid(i));
        w_ig = w(index_ig);
        X_ig = X(:,index_ig);
        
        gamma = w_ig'*w_ig + epsilon;
        sum_w = sum_w + gamma^(1/2);
        grad(index_ig) = X_ig'*(X*w-y) + 0.5*lambda*w_ig*gamma^(-1/2);
    end
    cost = 0.5*(y-X*w)'*(y-X*w) + lambda*sum_w;
end
