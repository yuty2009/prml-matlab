clc
clear

N = 4;
M = 6;
A = diag(abs(randn(N,1)));
B = randn(N,M);

[u0, s0, v0] = svd(B);

[u1, d1] = eig(B'*A*B);
diag(d1)'

d2 = myeig(diag(sqrt(diag(A)))*B);
diag(d2)'

[u3, d3] = myeigex(u0,s0,A);
diag(d3)'