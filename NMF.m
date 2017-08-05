%% Non-negative matrix factorization
% input params
% X: input matrix
% r: rank of the factorized matrix
% maxit: maximum iteration number
% output params
% W: feature matrix
% H: coefficient matrix
function [W H] = NMF(X, r, maxit)

[N, M] = size(X);

W = rand(N, r);
W = W./(ones(N,1)*sum(W));
H = rand(r, M);

it = 1;
while it < maxit
    H = H.*(W'*(X./(W*H)));
    W = W.*((X./(W*H))*H');
    W = W./(ones(N,1)*sum(W));
    it = it + 1;
end
