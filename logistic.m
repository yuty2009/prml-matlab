%% Logistic regression for binary classification (Page 205-208 of PRML)
% Iterative reweighted least square (IRLS) by Newton-Raphson
% iterative optimization scheme.
% w_new = w_old - (PHI'*R*PHI)^(-1)*PHI'*(y-t);
% 
% X: N by P design matrix with N samples of M features
% t: N by 1 target values {0,1}
% b: P by 1 weight vector
% b0: bias
function [varargout] = logistic(t,X,lambda)

if nargin < 3, lambda = 1e-4; end

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N, P] = size(PHI);
t(find(t==-1)) = 0; % the class label should be [1 0]

% initialization
w = zeros(P,1); % rough initialization
w(1) = log(mean(t)/(1-mean(t)));

% stop conditions
d_w = Inf;
maxit = 500;
stopeps = 1e-6;

i = 1;
while (d_w > stopeps)  && (i < maxit)
    
    wold = w;
    
    y = 1./(1+exp(-PHI*w)); % predicted target value
    R = diag(y.*(1-y)); % the variance matrix of target value
    % update with a norm2 regularization of w
    % H = PHI'*R*PHI + lambda*eye(P);
    if (P>N)
        invH = woodburyinv(lambda*eye(P), PHI', PHI, R);
    else
        invH = (lambda*eye(P) + PHI'*R*PHI)^(-1);
    end
    
    w = w - invH*(PHI'*(y-t)+lambda*w);
    
    d_w = norm(wold-w);
    
    disp(['Iteration ' num2str(i) ': wchange = ' num2str(d_w)]);
    i = i + 1;

end

if(i >= maxit)
    disp(['Optimization finished with maximum iterations = ' num2str(maxit)]);
end

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