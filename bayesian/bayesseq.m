%% Sequential learning Bayesian estimate of Gaussian posterior
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% b: P by 1 regression coefficients
% b0: the intercept
function [varargout] = bayesseq(y, X)

PHI = cat(2, ones(size(X,1),1), X); % add a constant column to cope with bias
[N, P] = size(PHI);

alpha = 2;
beta = 10;
w = zeros(P,1);
d_w = Inf;
rmse = Inf;
maxit = 5000;
stopeps = 0.001;

invSigma = alpha*eye(P);

i = 1;
while (rmse > stopeps)  && (i <= maxit)
    wold = w;
    invSigmaold = invSigma;
    
    n = randi(N);
    phi = PHI(n,:)';
    invSigma = invSigmaold + beta*phi*phi';
    w = (invSigma)^(-1)*(invSigmaold*wold + beta*phi*y(n));
    
    d_w = norm(wold - w);
    rmse = norm(PHI*w - y);
    
    fprintf('Iteration %i, sample = %i: rmse = %f, wchange = %f\n', i, n, rmse, d_w);
    i = i + 1;
end

disp(['Optimization terminated after ' num2str(i-1) ' iterations']);

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