%% Bayesian PCA
%  X: N observation by P variables
%  The model is:
%   z ~ N(0,I) % unit covariance to eliminate scaling indeterminancy
%   n ~ N(0,sigma^2*I) % n is independent of the latent variable z
%   x = W*z + mu + n
%   Psi is diagnal
%  Optimization via EM algorithm according to PRML Page 578
function varargout = bardpca(X)

[N,P] = size(X);

% subtract off the mean for each observation
mu = mean(X)';
X0 = X - repmat(mu',N,1);

% initialize W randomly
alphas = 2*ones(P,1);
sigma2 = 0.01;
EW = randn(P);

maxvalue = 1e9;
stopeps = 1e-6;
d_W = Inf;
maxit = 500;
it = 1;
while (d_W > stopeps)  && (it < maxit)
    Wold = EW;
%     % pruning alphas that larger than maxvalue
%     % which results in M non-zero columns of W
%     index0 = find(alphas > maxvalue);
%     index1 = setdiff(1:P, index0);
%     alphas1 = alphas(index1); % M
%     W1 = W(:,index1); % P by M
    
    % E step, calculate posterior
    invM = (EW'*EW + eye(P)*sigma2)^(-1); % M by M
    EZ = invM*EW'*X0'; % M by N
    EZZ = N*sigma2*invM + EZ*EZ'; % M by M
    EW = X0'*EZ'*(EZZ + sigma2*diag(alphas))^(-1); % P by M

    % M step, maximize the expectation
    
    sigma2 = (1/(N*P))*trace(X0'*X0 - 2*EW*EZ*X0 + EW*EZZ*EW');
    alphas = P./diag(EW'*EW);
    
    d_W = EW - Wold;
    d_W = norm(d_W)/norm(Wold);
    
    fprintf('Iteration %i: wchange = %f, max(alpha) = %f, min(alpha) = %f, sigma2 = %f\n', ...
        it, d_W, max(alphas), min(alphas), sigma2);
    it = it + 1;
end

% orthogonalize W to get a unique solution (PRML pp. 575
varargout{1} = EW; % orth(W); 
if nargout > 1, varargout{2} = mu; end
if nargout > 2, varargout{3} = sigma2; end
if nargout > 3, varargout{4} = alphas; end
