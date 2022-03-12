%% Compute the eigenvalues of (B'*A*B) from the svd of matrix B
% B is a N by M design matrix and A is a N by N diagonal matrix
% [U0 S0 V0] = svd(B)
% ev are the eigenvalues of (B'*A*B)
% the computation is efficient when M >> N
function [U1, EV] = myeigex(U0, S0, A)

[N, M] = size(S0); % S0 has the same size as B

if N > M
    disp('No. of rows should be larger than that of columns of B');
    return;
end

P = N;
D0 = S0(1:P, 1:P);
[U1, D1] = eig(D0*U0'*A*U0*D0);

EV = zeros(M);
EV(1:N,1:N) = D1;