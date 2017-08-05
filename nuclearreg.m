%% Least square regression with grouped lasso regularization (l_2 norm)
% The optimization objective is as following:
% w = {arg min}_w (||Xw-y||_2^2 + \lambda \sum_g^G ||w_{I_g}||_2)
% 
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% NG: X can be reshape to be N by NS (P/NG) by NG 
% b: P by 1 regression coefficients
% b0: the intercept
function [varargout] = nuclearreg(y,X,NG,lambda)

if nargin <= 3
    lambda = 1e-4;
end

PHI = cat(2, ones(size(X,1),1), X); % add a constant column to cope with bias
[N, P] = size(PHI);
NS = size(X,2)/NG; % 
assert(rem(NS, 1) == 0, 'samples in group not integer');
    
cvx_quiet(false);
cvx_begin
    variable W(NS,NG);
    variable b0(1,1);
    variable z(N,1);
    
    minimize norm(y-z) + lambda*norm_nuc(W); 
    subject to
        PHI*[W(:);b0] == z;
cvx_end

b = W(:);

if nargout == 1
    model.b = b;
    model.b0 = b0;
    varargout{1} = model;
elseif nargout == 2
    varargout{1} = b;
    varargout{2} = b0;
end
