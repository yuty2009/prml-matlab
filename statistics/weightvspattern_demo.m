%% Forward and backward model example
% Reference: On the interpretation of weight vectors of linear models in
% multivariate neuroimaging, Neuroimage, 2014

clc
clear

M = 2;
N1 = 100;
N2 = 100;
MU1 = [1.5;0];
MU2 = [-1.5;0];
SIGMA = [1.02,-0.30;-0.30,0.15];
X1 = mvnrnd(MU1,SIGMA,N1);
X2 = mvnrnd(MU2,SIGMA,N2);

X = cat(1, X1, X2);
y = [ones(N1,1); -1*ones(N2,1)];

% weights
[w1,b1] = FLDA(y,X);
[w2,b2] = logistic(y,X);

% patterns
y1 = X*w1+b1;
y2 = X*w2+b2;
a1 = X'*y1/(y1'*y1);
a2 = X'*y2/(y2'*y2);

t = min(X(:,1)):0.1:max(X(:,1));
v1 = (-w1(1)*t-b1)/w1(2);
v2 = (-w2(1)*t-b2)/w2(2);

figure;
hold on;
scatter(X1(:,1),X1(:,2),'bx');
scatter(X2(:,1),X2(:,2),'ro');
plot(t,v1,'g-');
plot(t,v2,'k-');
axis equal;
grid on;
legend('c1','c2','LDA','logistic');
