%% Least square multi-class classifier
% X: N by P feature matrix, N number of samples, P number of features
% t: N by 1 class labels (t=k indicate belong to class k)
% W: P by K regression coefficients
function [W] = lsclsm(t,X,lambda)

if nargin <= 2
    lambda = 1e-4;
end

PHI = cat(2,ones(size(X,1),1),X); % add a constant column to cope with bias
[N,P] = size(PHI);

K = length(unique(t));
T = full(sparse(1:N,t,1));

W = inv(PHI'*PHI+lambda*eye(P))*PHI'*T;