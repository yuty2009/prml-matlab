clc
clear
addpath(genpath('D:\research\document\machinelearning\Pattern_Recognition_and_Machine_Learning\PRML-master'))

D = 10;
N = 300;
MU = zeros(D,1);
SIGMA = diag([0.2,1.0,0.2,1.0,0.2,0.2,1.0,0.2,0.2,0.2]);
X = mvnrnd(MU,SIGMA,N);

% PC11 = PCA(X);
% PC12 = pca(X', D);
% PC21 = PPCA(X, D);
% PC22 = ppcaEm(X', D);
PC31 = bardpca(X);
model = ppcaVb(X', D); PC32 = model.W;

% hinton(PC11);
% hinton(PC12);
% hinton(PC21);
% hinton(PC22);
hinton(PC31);
hinton(PC32);