clc
clear

mu1 = [2 2];
mu2 = [4 4];
mu3 = [6 2];
Sigma = 0.25*[1 0; 0 1];

N1 = 50;
N2 = 50;
N3 = 50;
X1 = mvnrnd(mu1,Sigma,N1);
X2 = mvnrnd(mu2,Sigma,N2);
X3 = mvnrnd(mu3,Sigma,N3);

N = N1+N2+N3;
X = cat(1,X1,X2,X3);
y = [ones(N1,1);2*ones(N2,1);3*ones(N3,1)];
perm = randperm(N);
X = X(perm,:);
y = y(perm);

margin = 1;
xmin = min(X(:,1))-margin;
xmax = max(X(:,1))+margin;
ymin = min(X(:,2))-margin;
ymax = max(X(:,2))+margin;

figure;

%% Least square classifier
W = lsclsm(y,X);

% calculate separate line according to PRML Page 183 (4.10)
xa = [xmin;xmax];
ya1 = (-(W(2,1)-W(2,2))'*xa - (W(1,1)-W(1,2)))/(W(3,1)-W(3,2)); % 1 vs 2
ya2 = (-(W(2,1)-W(2,3))'*xa - (W(1,1)-W(1,3)))/(W(3,1)-W(3,3)); % 1 vs 3
ya3 = (-(W(2,2)-W(2,3))'*xa - (W(1,2)-W(1,3)))/(W(3,2)-W(3,3)); % 2 vs 3

subplot(221);
hold on;
scatter(X1(:,1),X1(:,2),'r');
scatter(X2(:,1),X2(:,2),'g');
scatter(X3(:,1),X3(:,2),'b');
plot(xa,ya1,'r');
plot(xa,ya2,'g');
plot(xa,ya3,'b');
axis([xmin xmax ymin ymax]);
legend('c1','c2','c3','1v2','1v3','2v3','Location','NW');
title('Least square');

%% Fisher's LDA classifier (one-versus-the-rest)
W = FLDAM(y,X,'1vR');

xa = [xmin;xmax];
ya1 = (-W(2,1)'*xa - W(1,1))/W(3,1);
ya2 = (-W(2,2)'*xa - W(1,2))/W(3,2);
ya3 = (-W(2,3)'*xa - W(1,3))/W(3,3);

subplot(222);
hold on;
scatter(X1(:,1),X1(:,2),'r');
scatter(X2(:,1),X2(:,2),'g');
scatter(X3(:,1),X3(:,2),'b');
plot(xa,ya1,'r');
plot(xa,ya2,'g');
plot(xa,ya3,'b');
axis([xmin xmax ymin ymax]);
legend('c1','c2','c3','1vR','2vR','3vR','Location','NW');
title('FLDA one-versus-the-rest');

%% Fisher's LDA classifier (one-versus-one)
W = FLDAM(y,X,'1v1');

xa = [xmin;xmax];
ya1 = (-W(2,1)'*xa - W(1,1))/W(3,1);
ya2 = (-W(2,2)'*xa - W(1,2))/W(3,2);
ya3 = (-W(2,3)'*xa - W(1,3))/W(3,3);

subplot(223);
hold on;
scatter(X1(:,1),X1(:,2),'r');
scatter(X2(:,1),X2(:,2),'g');
scatter(X3(:,1),X3(:,2),'b');
plot(xa,ya1,'r');
plot(xa,ya2,'g');
plot(xa,ya3,'b');
axis([xmin xmax ymin ymax]);
legend('c1','c2','c3','1v2','1v3','2v3','Location','NW');
title('FLDA one-versus-one');

%% Softmax regression
W = softmax(y,X);

xa = [xmin;xmax];
ya1 = (-(W(2,1)-W(2,2))'*xa - (W(1,1)-W(1,2)))/(W(3,1)-W(3,2)); % 1 vs 2
ya2 = (-(W(2,1)-W(2,3))'*xa - (W(1,1)-W(1,3)))/(W(3,1)-W(3,3)); % 1 vs 3
ya3 = (-(W(2,2)-W(2,3))'*xa - (W(1,2)-W(1,3)))/(W(3,2)-W(3,3)); % 2 vs 3

subplot(224);
hold on;
scatter(X1(:,1),X1(:,2),'r');
scatter(X2(:,1),X2(:,2),'g');
scatter(X3(:,1),X3(:,2),'b');
plot(xa,ya1,'r');
plot(xa,ya2,'g');
plot(xa,ya3,'b');
axis([xmin xmax ymin ymax]);
legend('c1','c2','c3','1v2','1v3','2v3','Location','NW');
title('Softmax regression');