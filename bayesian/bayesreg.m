%% Bayes linear regression
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% b: P by 1 regression coefficients
% b0: the intercept
function [varargout] = bayesreg(y, X)

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N, P] = size(PHI);

% initialize with least square estimation
if (P > N)
    invC = woodburyinv(1e-4*eye(P),PHI',PHI,eye(N));
else
    invC = (1e-4*eye(P)+ PHI'*PHI)^(-1);
end
w = invC*PHI'*y;
% w = ones(P,1); % rough initialization
alpha = 2;
beta = 10;

% stop conditions
d_w = Inf;
maxit = 500;
stopeps = 1e-6;

% [v,d] = eig(PHI'*PHI);
d = myeig(PHI);
d = diag(d);

i = 1;
while (d_w > stopeps)  && (i < maxit)
    wold = w;
    
    if (P>N)
        Sigma = woodburyinv(alpha*eye(P), PHI', PHI, (1/beta)*eye(N));
    else
        Sigma = (alpha*eye(P) + beta*PHI'*PHI)^(-1);
    end
    
    w = beta*Sigma*PHI'*y;
    gamma = sum(beta*d./(alpha+beta*d));
    alpha = gamma/(w'*w);
    rmse = sum((y-PHI*w).^2);
    beta  = max(N-gamma,eps)/(rmse+1e-32);
    
    evidence = (P/2)*log(alpha) + (N/2)*log(beta) - ...
        (beta/2)*rmse - (alpha/2)*w'*w - ...
        (1/2)*sum(log((beta*d+alpha))) - (N/2)*log(2*pi);
    
    d_w = norm(w-wold)/(norm(wold)+1e-32);
    
    fprintf('Iteration %i: evidence = %f, wchange = %f, rmse = %f, alpha = %f, beta = %f\n', ...
        i, evidence, d_w, rmse, alpha, beta);
    i = i + 1;
end

if(i < maxit)
    fprintf('Optimization of alpha and beta successfull.\n');
else
    fprintf('Optimization terminated due to max iteration.\n');
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
