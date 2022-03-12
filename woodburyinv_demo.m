clc
clear

N = 200;
M = 4000;

A = diag(randn(M,1));
B = randn(M,N);
C = B';
D = randn(N,N);

T = 1;

disp('Calculate inverse with Woodbury identity');
tic
for i = 1:T
    R1 = woodburyinv(A,B,C,D);
end
toc

disp('Calculate inverse directly');
tic
for i = 1:T
    R2 = (A+B*D^(-1)*C)^(-1);
end
toc
