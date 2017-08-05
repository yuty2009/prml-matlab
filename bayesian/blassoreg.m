%% Bayes linear regression with Laplace priors
% Laplace prior is represented by a scale mixture of normals
% i.e., with an exponential mixing density
% In this case, Jeffreys' prior is used instead of the exponential
%
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% b: P by 1 regression coefficients
% b0: the intercept
%
% Reference: M.A.T. Figueiredo, Adaptive sparseness for supervised
% learning, IEEE Trans Pattern Anal Mach Intell, vol. 25, (no. 9), pp.
% 1150-1159, 2003.
function [varargout] = blassoreg(y, X)

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N,P] = size(PHI);

% initialize with least square estimation
if (P > N)
    invC = woodburyinv(eye(P),PHI',PHI,eye(N));
else
    invC = (eye(P)+ PHI'*PHI)^(-1);
end
w = invC*PHI'*y;
% w = ones(P,1); % rough initialization
alphas = 2*ones(P,1);
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
    
    % E step
    if (P > N)
        Sigma = woodburyinv(diag(alphas), PHI', PHI, (1/beta)*eye(N));
    else
        Sigma = (diag(alphas) + beta*PHI'*PHI)^(-1);
    end
    w = beta*Sigma*PHI'*y;
    
    % M step
    alphas = 1./(w.^2+1e-32); % convergence is fast but not stable
    % alphas = 1./(w.^2+diag(Sigma)+1e-32);
    rmse = sum((y-PHI*w).^2);
    beta = N/(rmse+1e-32);
    
    evidence = (1/2)*sum(log(alphas)) + (N/2)*log(beta) - ...
        (beta/2)*rmse - (1/2)*w'*diag(alphas)*w - ...
        (1/2)*sum(log(beta*d+alphas)) - (N/2)*log(2*pi);
    
    d_w = norm(w-wold)/(norm(wold)+1e-32);
    
    fprintf(['Iteration %i: evidence = %f, wchange = %f, ' ...
        'rmse = %f, beta = %f\n'], ...
        i, evidence, d_w, rmse, beta);
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
