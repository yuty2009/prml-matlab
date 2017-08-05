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
function [varargout] = lassogroupreg2(y,X,group,lambda)

if nargin < 4, lambda = 1e-4; end

PHI = cat(2, ones(size(X,1),1), X); % add a constant column to cope with bias
[N, P] = size(PHI);

group = [0;group(:)]; % account for bias
groupid = unique(group);
NG1 = length(groupid);

epsilon = 1e-4;
w = ones(P,1);
d_w = Inf;
error = Inf;
d_error = Inf;
nta = 0.01;
maxit = 500;
stopeps = 0.001;

i = 1;
while (d_w > stopeps)  && (i < maxit)
    errorold = error;
    wold = w;
    
    for g = 1:NG1
        index_ig = find(group == groupid(g));
        w_ig = w(index_ig);
        PHI_ig = PHI(:,index_ig);
        
        gamma = w_ig'*w_ig + epsilon;
        grad = PHI_ig'*(PHI*w-y) + 0.5*lambda*w_ig*gamma^(-1/2);
        hess = PHI_ig'*PHI_ig + 0.5*lambda*gamma^(-1/2) ...
            - 0.25*lambda*w_ig*gamma^(-3/2)*w_ig';
        w_ig = w_ig - nta*(hess+1e-4*eye(size(hess)))^(-1)*grad;
        
        w(index_ig) = w_ig;
    end
    
    error = 0.5*(y-PHI*w)'*(y-PHI*w) + lambda*(w'*w+epsilon)^(1/2);
    
    d_w = norm(w-wold);
     
    fprintf('Iteration %i: error = %f, wchange = %f\n', ...
        i, error, d_w);
    i = i + 1;
end

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