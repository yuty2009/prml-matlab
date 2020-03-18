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
mu = mean(X);
X0 = X - repmat(mu,N,1);

% initialize W randomly
alphas = 2*ones(P,1);
W = randn(P);
sigma2 = 1/randg;

RXX = dot(X0(:),X0(:)); % total norm of X
maxvalue = 1e9;
stopeps = 1e-8;
d_W = Inf;
maxit = 500;
it = 1;
while (d_W > stopeps)  && (it < maxit)
    Wold = W;
    % pruning alphas that larger than maxvalue
    % which results in M non-zero columns of W
    index0 = find(alphas > maxvalue);
    index1 = setdiff(1:P, index0);
    m = length(index1);
    alphas1 = alphas(index1); % m
    W1 = W(:,index1); % P by m
    
    % E step, calculate posterior
    invM = (W1'*W1 + eye(m)*sigma2)^(-1); % m by m
    EZ = invM*W1'*X0'; % m by N
    EZZ = N*sigma2*invM + EZ*EZ'; % m by m
    W1 = X0'*EZ'*(EZZ + sigma2*diag(alphas1))^(-1); % P by m
    W(:,index1) = W1;

    % M step, maximize the expectation
    U = chol(EZZ); % m by m
    WR = W1*U'; % P by m
    WX = W1'*X0'; % m by N
    sigma2 = (RXX - 2*dot(EZ(:),WX(:)) + dot(WR(:),WR(:)))/(N*P); % P by P (12.57)
    alphas1 = P./diag(W1'*W1);
    alphas(index1) = alphas1;
    
    d_W = W - Wold;
    d_W = norm(d_W)/norm(Wold);
    
    fprintf('Iteration %i: wchange = %f, max(alpha) = %f, min(alpha) = %f, sigma2 = %f\n', ...
        it, d_W, max(alphas), min(alphas), sigma2);
    it = it + 1;
end

% orthogonalize W to get a unique solution (PRML pp. 575
varargout{1} = W; % orth(W); 
if nargout > 1, varargout{2} = mu; end
if nargout > 2, varargout{3} = sigma2; end
if nargout > 3, varargout{4} = alphas; end
