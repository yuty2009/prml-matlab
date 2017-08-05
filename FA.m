%% Factor Analysis
%  X: N observation by P variables
%  The model is:
%   z ~ N(0,I) % unit covariance to eliminate scaling indeterminancy
%   n ~ N(0,Psi) % n is independent of the latent variable z
%   x = W*z + mu + n
%   Psi is diagnal
%  Optimization via EM algorithm according to PRML Page 584-586
function varargout = FA(X)

[N,P] = size(X);

% subtract off the mean for each observation
mu = mean(X)';
X0 = X - repmat(mu',N,1);

% initialize W and Psi with PCA results
% refer to PRML Page 574-575
epsilon = 1e-4;
[PC,EV] = PCA(X);
diagEV = diag(EV);
Psi = epsilon*eye(P);
W = PC*sqrt(diag(diagEV));

stopeps = 1e-6;
d_W = Inf;
maxit = 500;
it = 1;
while (d_W > stopeps)  && (it < maxit)
    Wold = W;
    % E step, calculate posterior
    invPsi = diag(1./diag(Psi));
    invM = (eye(NC) + W'*invPsi*W)^(-1); % NC by NC
    EZ = invM*W'*invPsi*X0'; % NC by N
    EZZ = N*invM + EZ*EZ'; % NC by NC

    % M step, maximize the expectation
    W = X0'*EZ'*EZZ^(-1); % P by NC
    Psi = diag(diag((1/N)*(X0'*X0 - W*EZ*X0))); % P by P
    
    d_W = W - Wold;
    d_W = norm(d_W)/norm(Wold);
    
    fprintf('Iteration %i: wchange = %f\n', it, d_W);
    it = it + 1;
end


varargout{1} = W;
if nargout > 1, varargout{2} = mu; end
if nargout > 2, varargout{3} = sigma2; end
