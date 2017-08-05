clc
clear

N = 20;
X(:,1) = 5*randn(N,1);
X(:,2) = X(:,1) + 2*randn(N,1);
mX = mean(X,1); 
X0 = X - repmat(mX,N,1);

margin = 2;
xmin = min(X(:,1))-margin;
xmax = max(X(:,1))+margin;
ymin = min(X(:,2))-margin;
ymax = max(X(:,2))+margin;

PC = PCA(X);

rdim = 1;
W_pca = PC(:,1:rdim);
X_pca = X0*W_pca;
X1 = X0*PC;
X2 = X1*PC';

t = [xmin xmax];
v1 = W_pca(2)/W_pca(1)*(t-mX(1)) + mX(2);

X_recon(:,1) = X_pca*W_pca(1)/norm(W_pca) + mX(1);
X_recon(:,2) = X_pca*W_pca(2)/norm(W_pca) + mX(2);

figure;
subplot(121);
hold on;
scatter(X(:,1),X(:,2),'bx');
scatter(X_recon(:,1),X_recon(:,2),'ro');
plot(t,v1,'g-');
axis([xmin,xmax,ymin,ymax]);
legend('input','reconstructed','PCA direction','Location','NW');

subplot(122);
hold on;
scatter(X2(:,1),X2(:,2),'bx');
scatter(X1(:,1),X1(:,2),'ro');
axis([xmin,xmax,ymin,ymax]);
title('rotated data');