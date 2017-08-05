clc
clear

K = 16;
N = 1000*K;
X = mvnrnd([0 0], eye(2), N);

[C, Q] = LBGVQ(X, K, 0.001);

figure;
hold on;
scatter(X(:,1),X(:,2),1,Q);
scatter(C(:,1),C(:,2),100,'k','*');