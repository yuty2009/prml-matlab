%% Compute the eigenvalues of X'*X by SVD
% X is a N by P design matrix
function d = myeig(X)

[N,P] = size(X);

S = svd(X);
d1 = S.^2;
M = min(N,P);
d = zeros(P,1);
d(1:M) = d1(1:M);
d = diag(d);