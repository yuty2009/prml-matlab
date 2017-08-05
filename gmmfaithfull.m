clc
clear

% N = 500;
% K = 3;
% MU = [3 2;0 -1;-3 -5];
% SIGMA = cat(3,[1 0;0 4],[4 0;0 1],[1 0;0 4]);
% Pz = ones(1,K)/K;
% X = gmmrnd(N, MU, SIGMA, Pz);

load('faithfull.txt');
X = faithfull(:,2:3);
z = faithfull(:,1);
[N,P] = size(X);
for i = 1:P
    X(:,i) = zscore(X(:,i));
end

K = 15;
opts.verbose = 1;
opts.plot = 1;
gmmmodel = gmmfitvb(X,K,opts);

% figure
% plot(gmm.L)
