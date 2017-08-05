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
% MU = [1 2;-3 -5];
% SIGMA = cat(3,[2 0;0 .5],[1 0;0 1]);
% Pz = ones(1,2)/2;
% 
% x = [-10:0.2:10];
% y = [-10:0.2:10];
% Px = zeros(length(x),length(y));
% for k = 1:length(x)
%     for j = 1:length(y)
%         Px(k,j) = gmmpdf([x(k) y(j)], MU, SIGMA, Pz);
%     end
% end
% mesh(x, y, Px);

function Px = gmmpdf(X, MU, SIGMA, Pz)

N = size(X,1);
[K D] = size(MU);

Pz = abs(Pz);
Pz = Pz./sum(Pz);

Pxgz = zeros(N,K);
for k = 1:K
    Pxgz(:,k) = mvnpdf(X, MU(k,:), SIGMA(:,:,k));
end

Px = Pxgz*Pz';