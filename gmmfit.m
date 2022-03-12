%% Fitting the Gaussian mixture model give observations X with the
%% Expectation Maximization (EM) algorithm
% input param
% X: X is a N-by-D matrix specifying N D-dimensional samples from Gaussian mixture model
% MU: MU is a k-by-d matrix specifying the d-dimensional mean of each of the k
% components.
% SIGMA: SIGMA specifies the covariance of each component.
% The size of SIGMA is d-by-d-by-k. In this case, SIGMA(:,:,I) is the
% covariance of component I.
% Pz: Pz is an 1-by-k vector specifying the mixing proportions of each
% component. If pz does not sum to 1, the function normalizes it.

function [MU,SIGMA,Pz] = gmmfit(X,K,opts)

maxit = 500;
stopeps = 1e-6;
if nargin > 2
    if isfield(opts,'maxit')
        maxit = opts.maxit;
    end
    if isfield(opts,'stopeps')
        stopeps = opts.stopeps;
    end
else
    opts.verbose = 0;
end

[N,D] = size(X);

% random initialization
% MU = X(1:K,:);
% SIGMA = reshape(repmat(eye(D),1,K),D,D,K);
% Pz = ones(1,K)/K;

% initialization with kmeans
[IDX,MU] = kmeans(X,K);
for k = 1:K
    idx1 = find(IDX==k);
    X1 = X(idx1,:);
    SIGMA(:,:,k) = cov(X1);
    Pz(k) = length(idx1)/N;
end

L = -Inf;
for step = 1:maxit
    Lold = L;
    % E step
    [dummy,r] = gmmposterior(X, MU, SIGMA, Pz);
    % M step
    NK = sum(r) + 1e-10;
    % MU = (r'*X)./repmat(NK',1,D);
    for k = 1:K
        MU(K,:) = r(:,k)'*X/NK(k);
        diff = X - repmat(MU(k,:),N,1);
        SIGMA(:,:,k) = diff'*diag(r(:,k))*diff/NK(k);
    end
    Pz = NK/N;
    % log likelihood
    Pxgz = zeros(N,K);
    for k = 1:K
        Pxgz(:,k) = mvnpdf(X, MU(k,:), SIGMA(:,:,k));
    end
    Px = Pxgz*Pz';
    L = sum(log(Px));
    
    dL = abs(L-Lold);
    if dL < stopeps
        break;
    end
    
    if isfield(opts,'verbose') && opts.verbose == 1
        disp(['Optimization step ' num2str(step), ', log-likelihood = ' num2str(L)]);
    end
end
