%% Posterior probability of Gaussian mixture model give observations X
% input param
% X: X is a N-by-D matrix specifying N D-dimensional samples from Gaussian mixture model
% MU: MU is a k-by-d matrix specifying the d-dimensional mean of each of the k
% components.
% SIGMA: SIGMA specifies the covariance of each component.
% The size of SIGMA is d-by-d-by-k. In this case, SIGMA(:,:,I) is the
% covariance of component I.
% Pz: Pz is an 1-by-k vector specifying the mixing proportions of each
% component. If pz does not sum to 1, the function normalizes it.
%
% example:
%
% N = 500;
% MU = [1 2;-3 -5];
% SIGMA = cat(3,[2 0;0 .5],[1 0;0 1]);
% Pz = ones(1,2)/2;
% 
% X = gmmrnd(N, MU, SIGMA, Pz);
% [z Pzmax] = gmmposterior(X, MU, SIGMA, Pz);
% 
% figure;
% index1 = find(z==1);
% index2 = find(z==2);
% scatter(X(index1,1),X(index1,2),'r.');
% hold on;
% scatter(X(index2,1),X(index2,2),'b.');

function [z,Pzgx] = gmmposterior(X, MU, SIGMA, Pz)

N = size(X,1);
[K,D] = size(MU);

Pz = abs(Pz);
Pz = Pz./sum(Pz);

Pxgz = zeros(N,K);
for k = 1:K
    Pxgz(:,k) = mvnpdf(X, MU(k,:), SIGMA(:,:,k));
end
Px = Pxgz*Pz';

Pxaz = zeros(N,K);
for k = 1:K
    Pxaz(:,k) = Pxgz(:,k)*Pz(k);
end

Pzgx = zeros(N,K);
for k = 1:K
    for n = 1:N
        Pzgx(n,k) = Pxaz(n,k)/Px(n);
    end
end

z = zeros(size(X,1),1);
Pzmax = zeros(size(X,1),1);
for n = 1:N
    [postp,index] = max(Pzgx(n,:));
    z(n) = index;
    Pzmax(n) = postp;
end