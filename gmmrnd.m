%% Generate N D-dimensional samples from Gaussian mixture model
% input param
% N: size(X,1)
% MU: MU is a k-by-d matrix specifying the d-dimensional mean of each of the k
% components.
% SIGMA: SIGMA specifies the covariance of each component.
% The size of SIGMA is d-by-d-by-k. In this case, SIGMA(:,:,I) is the
% covariance of component I.
% Pz: Pz is an 1-by-k vector specifying the mixing proportions of each
% component. If pz does not sum to 1, the function normalizes it.
%
% example:
% MU = [1 2;-3 -5];
% SIGMA = cat(3,[2 0;0 .5],[1 0;0 1]);
% Pz = ones(1,2)/2;
% N = 500;
% X = gmmrnd(N, MU, SIGMA, Pz);
% scatter(X(:,1),X(:,2),'.');

function X = gmmrnd(N, MU, SIGMA, Pz)

[K D] = size(MU);

for n = 1:N
    
    % generate z from Pz
    index = randptable(Pz);
    
    % generate data from the index-th gaussian component
    X(n,:) = mvnrnd(MU(index,:), SIGMA(:,:,index));
    
end