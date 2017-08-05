clc
clear

D = 10;
N = 300;
MU = zeros(D,1);
SIGMA = diag([0.1,1,0.1,1,0.1,0.1,1,0.1,0.1,0.1]);
X = mvnrnd(MU,SIGMA,N);

PC1 = PCA(X);
PC2 = PPCA(X);
PC3 = BPCA(X);

hinton(PC1);
hinton(PC2);
hinton(PC3);