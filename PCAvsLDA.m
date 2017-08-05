clc
clear

N1 = 100;
N2 = 100;
mu1 = [1 3];
mu2 = [3 1];
Sigma = [1 1.5; 1.5 3];
X1 = mvnrnd(mu1,Sigma,N1);
X2 = mvnrnd(mu2,Sigma,N2);
N = N1 + N2;
X = cat(1, X1, X2);
y = [ones(N1,1); -1*ones(N2,1)];
mX = mean(X,1);

PC = PCA(X);
rdim = 1;
W_pca = PC(:,1:rdim);

[W_lda, b_lda] = FLDA(y,X);

t = [min(X(:,1)):0.1:max(X(:,1))];
v1 = W_pca(2)/W_pca(1)*(t-mX(1)) + mX(2);
v2 = W_lda(2)/W_lda(1)*(t-mX(1)) + mX(2);
v3 = (-W_lda(1)*t - b_lda)/W_lda(2); % separate plane

figure;
hold on;
scatter(X1(:,1), X1(:,2), 'bx');
scatter(X2(:,1), X2(:,2), 'ro');
plot(t, v1, 'r-');
plot(t, v2, 'g-');
plot(t, v3, 'b-');
axis equal;
legend('c1', 'c2', 'PCA main direction', 'LDA projection direction', 'LDA separate plane');