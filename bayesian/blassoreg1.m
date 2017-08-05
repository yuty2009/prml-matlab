%% Bayes linear regression with Laplace priors
% Laplace prior is represented by a scale mixture of normals
% i.e., with an exponential mixing density
%
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% b: P by 1 regression coefficients
% b0: the intercept
%
% Reference: 
% [1]. M.A.T. Figueiredo, Adaptive sparseness for supervised
% learning, IEEE Trans Pattern Anal Mach Intell, vol. 25, (no. 9), pp.
% 1150-1159, 2003.
% [2]. T. Park and G. Casella, The Bayesian Lasso, Journal of the American
% Statistical Association, vol. 103, (no. 482), 2008
% [3]. Ata Kaban, On Bayesian classification with Laplace priors, Pattern
% Recognition Letters, vol. 28, pp. 1271-1282, 2007
function [varargout] = blassoreg1(y, X)

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N,P] = size(PHI);

% initialize with least square estimation
if (P > N)
    invC = woodburyinv(1e-4*eye(P),PHI',PHI,eye(N));
else
    invC = (1e-4*eye(P)+ PHI'*PHI)^(-1);
end
w = invC*PHI'*y;
% w = ones(P,1); % rough initialization
lambda = P*sqrt(mean((y-PHI*w).^2))/sum(abs(w));
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
        Sigma = (diag(alphas) + (1/beta)*PHI'*PHI)^(-1);
    end
    w = beta*Sigma*PHI'*y;
    alphas = sqrt(max(lambda,eps)./(w.^2+diag(Sigma)+1e-32));
    
    % M step
    invalphas = 1/(lambda+1e-32) + 1./(alphas+1e-32);
    lambda = 2*P/(sum(invalphas)+1e-32);
    rmse = sum((y-PHI*w).^2);
    beta = N/(rmse+1e-32);
    
    evidence = (1/2)*sum(log(alphas)) + (N/2)*log(beta) - ...
        (beta/2)*rmse - (1/2)*w'*diag(alphas)*w - ...
        (1/2)*sum(log(beta*d+alphas)) - (N/2)*log(2*pi);
    
    d_w = norm(w-wold)/(norm(wold)+1e-32);
    
    disp(['Iteration ' num2str(i)  ': evidence = ' num2str(evidence) ...
        ', wchange = ' num2str(d_w) ', rmse = ' num2str(rmse) ...
        ', lambda = ' num2str(lambda)]);
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
