%% Probabilistic PCA
%  X: N observation by P variables
%  The model is:
%   z ~ N(0,I) % unit covariance to eliminate scaling indeterminancy
%   n ~ N(0,sigma^2*I) % n is independent of the latent variable z
%   x = W*z + mu + n
%   Psi is diagnal
%  Optimization via EM algorithm according to PRML Page 578
function varargout = PPCA(X)

[N,P] = size(X);

% subtract off the mean for each observation
mu = mean(X)';
X0 = X - repmat(mu',N,1);

% initialize W and sigma randomly
sigma2 = 0.01;
W = randn(P);

stopeps = 1e-6;
d_W = Inf;
maxit = 500;
it = 1;
while (d_W > stopeps)  && (it < maxit)
    Wold = W;
    % E step, calculate posterior
    invM = (W'*W + eye(P)*sigma2)^(-1); % NC by NC
    EZ = invM*W'*X0'; % NC by N
    EZZ = N*sigma2*invM + EZ*EZ'; % NC by NC

    % M step, maximize the expectation
    W = X0'*EZ'*EZZ^(-1); % P by NC
    sigma2 = (1/(N*P))*trace(X0'*X0 - 2*W*EZ*X0 + EZZ*W'*W); % P by P
    
    d_W = W - Wold;
    d_W = norm(d_W)/norm(Wold);
    
    fprintf('Iteration %i: wchange = %f, sigma2 = %f\n', it, d_W, sigma2);
    it = it + 1;
end

varargout{1} = W;
if nargout > 1, varargout{2} = mu; end
if nargout > 2, varargout{3} = sigma2; end
