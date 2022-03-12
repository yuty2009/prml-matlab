clc
clear

N = 1000;
MU = [0,0];
SIGMA = [1 1.5;1.5 4];
X = mvnrnd(MU,SIGMA,N);

var11 = var(X(:,1));
var12 = var(X(:,2));

[U1,V1] = eig(X'*X);

t = [min(X(:,1)) max(X(:,1))];
v11 = -U1(1,1)*t/U1(2,1);
v12 = -U1(1,2)*t/U1(2,2);

X_prj = X*U1;
var21 = var(X_prj(:,1));
var22 = var(X_prj(:,2));

[U2,V2] = eig(X_prj'*X_prj);

v21 = -U2(1,1)*t/U2(2,1);
v22 = -U2(1,2)*t/U2(2,2);

figure;
subplot(121);
hold on;
scatter(X(:,1),X(:,2));
plot(t, v11, 'g-');
plot(t, v12, 'r-');
error_ellipse(cov(X),mean(X));
axis([-6 6 -6 6]);
subplot(122);
hold on;
scatter(X_prj(:,1),X_prj(:,2));
plot(t, v21, 'g-');
plot(t, v22, 'r-');
error_ellipse(cov(X_prj),mean(X_prj));
axis([-6 6 -6 6]);
