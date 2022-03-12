%% Logistic regression for binary classification (Page 205-208 of PRML)
% Iterative reweighted least square (IRLS) by Newton-Raphson
% iterative optimization scheme.
% w_new = w_old - (PHI'*R*PHI)^(-1)*PHI'*(y-t);
% 
% X: N by P design matrix with N samples of M features
% t: N by 1 target values  of {1,-1}
% NG: X can be reshape to be N by NS (P/NG) by NG 
% b: P by 1 weight vector
% b0: bias
function [varargout] = nuclearlog(t,X,NG,lambda)

if nargin <= 3
    lambda = 1e-4;
end

% t(find(t==-1)) = 0; % the class label should be [1 0]

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N,P] = size(PHI);
NS = size(X,2)/NG; %
assert(rem(NS, 1) == 0, 'samples in group not integer');

cvx_quiet(false);
cvx_begin
    variable W(NS,NG);
    variable b0(1,1);
    variable z(N,1);
    
    %% the class label should be [1 0] for the following formular (Page
    %% 206, Equ. 4.90) % LASSO regularized
    % minimize sum(t.*log(1+exp(-z)) + (1-t).*(log(1+exp(-z))+z)) + lambda*norm(w,1);
    
    %% the class label must be [1 -1] for the simplified formular
    minimize sum(log(1+exp(-z))) + lambda*norm_nuc(W); 
    
    subject to
        % PHI*w == z;
        t.*(PHI*[W(:);b0]) == z;
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
