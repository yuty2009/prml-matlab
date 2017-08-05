%% Bayesian logistic regression for binary classification 
% (Page 353-356 of PRML)
% Optimization through Laplace approximation & IRLS.
% 
% X: N by P design matrix with N samples of M features
% t: N by 1 target values 
% w: P+1 by 1 weight vector
function [varargout] = bayeslog(t, X)

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N, P] = size(PHI);
t(find(t==-1)) = 0; % the class label should be [1 0]

% initialize with least square estimation
if (P > N)
    invC = woodburyinv(1e-4*eye(P),PHI',PHI,eye(N));
else
    invC = (1e-4*eye(P)+ PHI'*PHI)^(-1);
end
w = invC*PHI'*(t-mean(t));
% w = ones(P,1); % rough initialization
alpha = 1;

% stop conditions
d_w = Inf;
maxit = 500;
stopeps = 1e-6;

i = 1;
while (d_w > stopeps) && (i <= maxit)
    wold = w;
    
    %% E step (IRLS update)
    y = 1./(1+exp(-PHI*w)); % predicted target value
    diagR = y.*(1-y);
    R = diag(diagR); % the variance matrix of target value
    invR = diag(1./diagR);
    if (P>N)
        Sigma = woodburyinv(alpha*eye(P), PHI', PHI, invR);
    else
        Sigma = (alpha*eye(P) + PHI'*R*PHI)^(-1);
    end
    w = w - Sigma*(PHI'*(y-t)+alpha*w);
    
    %% M step
    % [v, d] = eig(PHI'*R*PHI);
    d = myeig(diag(sqrt(diagR))*PHI);
    d = diag(d);
    
    gamma = sum(d./(alpha+d)); 
    alpha = gamma/(w'*w);
    
    %% Calculate evidence
    evidence = sum(t.*log(y)+(1-t).*log(1-y)) + (P/2)*log(alpha) ...
        - (alpha/2)*(w'*w) - (1/2)*sum(log(alpha+d));
    
    d_w = norm(w-wold)/(norm(wold)+1e-32);
    
    fprintf('Iteration %i: evidence = %f, wchange = %f, alpha = %f\n', ...
        i, evidence, d_w, alpha);
    i = i + 1;
end

if(i < maxit)
    fprintf('Optimization of alpha and beta successfull.\n');
else
    fprintf('Optimization terminated due to max iteration.\n');
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
