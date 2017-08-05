clc
clear

mu1 = [2 4];
mu2 = [4 2];
Sigma = [1 1.5; 1.5 3];

N1 = 50;
N2 = 50;
X1 = mvnrnd(mu1,Sigma,N1);
X2 = mvnrnd(mu2,Sigma,N2);

N = N1+N2;
X = cat(1,X1,X2);
y = [ones(N1,1);-1*ones(N2,1)];
perm = randperm(N);
X = X(perm,:);
y = y(perm);

opts.lrate = 0.1;
[w,b] = perceptron(y,X);

xa = [min(X(:,1)) max(X(:,1))]';
ya = (-w(1)'*xa - b)/w(2);

figure;
hold on;
scatter(X1(:,1),X1(:,2),'r*');
scatter(X2(:,1),X2(:,2),'bo');
plot(xa,ya,'k');