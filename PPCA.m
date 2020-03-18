%% Probabilistic PCA
%  X: N observation by P variables
%  m: dimension of target space
%  The model is:
%   z ~ N(0,I) % unit covariance to eliminate scaling indeterminancy
%   n ~ N(0,sigma^2*I) % n is independent of the latent variable z
%   x = W*z + mu + n
%   Psi is diagnal
%  Optimization via EM algorithm according to PRML Page 578
function varargout = PPCA(X, m)

[N,P] = size(X);

% subtract off the mean for each observation
mu = mean(X);
X0 = X - repmat(mu,N,1);

% initialize W and sigma randomly
W = randn(P, m);
sigma2 = 1/randg;

RXX = dot(X0(:),X0(:)); % total norm of X
I = eye(m);
stopeps = 1e-8;
d_W = Inf;
maxit = 500;
it = 1;
while (d_W > stopeps)  && (it < maxit)
    Wold = W;
    % E step, calculate posterior
    invM = (W'*W + sigma2*I)^(-1); % m by m
    EZ = invM*W'*X0'; % m by N (12.54)
    EZZ = N*sigma2*invM + EZ*EZ'; % m by m (12.55)

    % M step, maximize the expectation
    W = X0'*EZ'/EZZ; % P by m (12.56)
    U = chol(EZZ); % m by m
    WR = W*U'; % P by m
    WX = W'*X0'; % m by N
    sigma2 = (RXX - 2*dot(EZ(:),WX(:)) + dot(WR(:),WR(:)))/(N*P); % P by P (12.57)
    
    d_W = W - Wold;
    d_W = norm(d_W)/norm(Wold);
    
    fprintf('Iteration %i: wchange = %f, sigma2 = %f\n', it, d_W, sigma2);
    it = it + 1;
end

% orthogonalize W to get a unique solution (PRML pp. 575
varargout{1} = W; % orth(W); 
if nargout > 1, varargout{2} = mu; end
if nargout > 2, varargout{3} = sigma2; end
