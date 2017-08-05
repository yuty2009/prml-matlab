%% Whitten the data to have identity covariance matrix
%  X: N by M data matrix with each column is a sample
%     and N < M
%  P: 
function [Y,P,invP] = whiten(X)

[N,M] = size(X);

% covX = cov(X');
% [U,D] = eig(covX);
% P = D^(-1/2)*U';
% invP = U*D^(1/2);

[U,S,V] = svd(X);
K = min([M,N]);
S1 = sqrt(1/(M-1))*S(1:K,1:K);
P = S1^(-1)*U';
invP = U*S1;

Y = P*X;