%% Bayesian PCA
%  X: N observation by P variables
%  The model is:
%   z ~ N(0,I) % unit covariance to eliminate scaling indeterminancy
%   n ~ N(0,sigma^2*I) % n is independent of the latent variable z
%   x = W*z + mu + n
%   Psi is diagnal
%  Optimization via EM algorithm according to PRML Page 578
function varargout = BPCA(X)

[N,P] = size(X);

% subtract off the mean for each observation
mu = mean(X)';
X0 = X - repmat(mu',N,1);

% initialize W randomly
alphas = 2*ones(P,1);
sigma2 = 0.01;
W = randn(P);

maxvalue = 1e9;
stopeps = 1e-6;
d_W = Inf;
maxit = 500;
it = 1;
while (d_W > stopeps)  && (it < maxit)
    Wold = W;
    % pruning alphas that larger than maxvalue
    index0 = find(alphas > maxvalue);
    index1 = setdiff(1:P, index0);
    alphas1 = alphas(index1);
    W1 = W(:,index1);
    
    % E step, calculate posterior
    invM = (W1'*W1 + eye(length(index1))*sigma2)^(-1); % P by P
    EZ = invM*W1'*X0'; % P by N
    EZZ = N*sigma2*invM + EZ*EZ'; % P by P

    % M step, maximize the expectation
    W1 = X0'*EZ'*(EZZ + sigma2*diag(alphas1))^(-1); % P by P
    sigma2 = (1/(N*P))*trace(X0'*X0 - 2*W1*EZ*X0 + EZZ*W1'*W1); % P by P
    
    alphas1 = P./diag(W1'*W1);
    alphas(index1) = alphas1;
    W(:,index1) = W1;
    
    d_W = W - Wold;
    d_W = norm(d_W)/norm(Wold);
    
    fprintf('Iteration %i: wchange = %f, sigma2 = %f\n', it, d_W, sigma2);
    it = it + 1;
end

varargout{1} = W;
if nargout > 1, varargout{2} = mu; end
if nargout > 2, varargout{3} = sigma2; end
if nargout > 3, varargout{4} = alphas; end
