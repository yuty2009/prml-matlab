clc
clear

trn = load('riply_trn');
X1 = trn.X';
y1 = trn.y';
y1(y1==1) = 1;
y1(y1==2) = -1;

tst = load('riply_tst');
X2 = tst.X';
y2 = tst.y';
y2(y2==1) = 1;
y2(y2==2) = -1;

xmin = -1.5;%min(X1(:,1));
xmax = 1;%max(X1(:,1));
ymin = -0.2;%min(X1(:,2));
ymax = 1.2;%max(X1(:,2));
[xx,yy] = meshgrid(xmin:0.02:xmax,ymin:0.02:ymax);
X3 = [xx(:),yy(:)];

opts.ktype = 'rbf';
opts.args = [0.5];
opts.lambda = 1e-4;
opts.method = 'blassoprobit';
% model = skFLDA(y1,X1,opts);
model = rvmtrain(y1,X1,opts);
yP1 = kpredict(X1,model);
yP2 = kpredict(X2,model);
yP3 = kpredict(X3,model);
yP1 = sign(yP1);
yP2 = sign(yP2);
yP3 = sign(yP3);

index11 = find(y1==1);
index12 = find(y1==-1);
index13 = find(yP1~=y1);
index21 = find(y2==1);
index22 = find(y2==-1);
index23 = find(yP2~=y2);
index31 = find(yP3==1);
index32 = find(yP3==-1);


figure;

subplot(221);
hold on;
scatter(X1(index11,1), X1(index11,2), 'bx');
scatter(X1(index12,1), X1(index12,2), 'ro');
scatter(X1(index13,1), X1(index13,2), 100, 'ko');
legend('c1', 'c2');
title(['train acc = ' num2str(length(index13)/length(y1))]);

subplot(222);
hold on;
scatter(X2(index21,1), X2(index21,2), 'bx');
scatter(X2(index22,1), X2(index22,2), 'ro');
scatter(X2(index23,1), X2(index23,2), 100, 'ko');
legend('c1', 'c2');
title(['test acc = ' num2str(length(index23)/length(y2))]);

subplot(223);
hold on;
scatter(X3(index31,1), X3(index31,2), 'b.');
scatter(X3(index32,1), X3(index32,2), 'r.');
legend('c1', 'c2');
title('decision boundary');

subplot(224);
hold on;
scatter(X1(index11,1), X1(index11,2), 'bx');
scatter(X1(index12,1), X1(index12,2), 'ro');
scatter(model.sv(:,1), model.sv(:,2), 100, 'ko');
legend('c1', 'c2');
title('support vectors');
