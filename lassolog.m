%% Logistic regression for binary classification (Page 205-208 of PRML)
% Iterative reweighted least square (IRLS) by Newton-Raphson
% iterative optimization scheme.
% w_new = w_old - (PHI'*R*PHI)^(-1)*PHI'*(y-t);
% 
% X: N by P design matrix with N samples of M features
% t: N by 1 target values  of [1 -1]
% w: P+1 by 1 weight vector
function [varargout] = lassolog(t,X,lambda)

if nargin < 3, lambda = 1e-4; end

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N, P] = size(PHI);
% t(find(t==-1)) = 0; % the class label should be [1 0]

% initialization
w = zeros(P,1); % rough initialization

cvx_quiet(false);
cvx_begin
    variable w(P,1);
    variable z(N,1);
    
    %% the class label should be [1 0] for the following formular (Page
    %% 206, Equ. 4.90) % LASSO regularized
    % minimize sum(t.*log(1+exp(-z)) + (1-t).*(log(1+exp(-z))+z)) +
    % lambda*norm(w,1);'
    
    %% the class label must be [1 -1] for the simplified formular
    minimize sum(log(1+exp(-z))) + lambda*norm(w,1); % LASSO regularized
    
    subject to
        % PHI*w == z;
        t.*(PHI*w) == z;
cvx_end

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