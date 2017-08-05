clc
clear

P = 1;
N1 = 1000;
N2 = 1000;
MU1 = -1;
MU2 = 1;
SIGMA = 1;
X1 = mvnrnd(MU1,SIGMA,N1);
X2 = mvnrnd(MU2,SIGMA,N2);
y1 = -1*ones(N1,1);
y2 = ones(N2,1);
X = cat(1,X1,X2);
y = cat(1,y1,y2);

yp = X;
[TPR,FPR,AUC] = ROC(y,yp);

figure;
subplot(121);
hold on;
hist(X1,100);
h = findobj(gca,'Type','patch'); 
set(h,'FaceColor','r') 
hist(X2,100);
subplot(122);
hold on;
plot([0,1],[0,1],'r--');
plot(FPR,TPR);
axis([0,1,0,1]);
