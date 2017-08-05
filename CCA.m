%% Canonical correlation analysis
%  Refer to canoncorr in MATLAB statiscal toolbox
%  X: P by N1 matrix
%  Y: P by N2 matrix
%  
function [A,B,r,U,V] = CCA(X,Y)

[P,N1] = size(X);
[P,N2] = size(Y);

XX = [X,Y];
SIGMA = cov(XX);
SIGMA11 = SIGMA(1:N1,1:N1);
SIGMA22 = SIGMA(N1+1:N1+N2,N1+1:N1+N2);
SIGMA12 = SIGMA(1:N1,N1+1:N1+N2);
SIGMA21 = SIGMA12';

SA = SIGMA11^(-1)*SIGMA12*SIGMA22^(-1)*SIGMA21;
SB = SIGMA22^(-1)*SIGMA21*SIGMA11^(-1)*SIGMA12;
[A0,DA] = eig(SA);
[B0,DB] = eig(SB);
DA = diag(DA);
DB = diag(DB);
[r0,idxA] = sort(DA,'descend');
[r0,idxB] = sort(DB,'descend');
A1 = A0(:,idxA);
B1 = B0(:,idxB);

MA = (A1'*SIGMA11*A1)^(-1/2);
MB = (B1'*SIGMA22*B1)^(-1/2);
A = A1*MA;
B = B1*MB;
r = sqrt(r0);
if nargout > 3
    U = X * A;
    V = Y * B;
end