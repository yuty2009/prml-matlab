clc
clear

datapath = 'e:\prmldata\bcw\';
% datapath = '/Users/n0n/work/data/prmldata/bcw/';

data = load([datapath 'breast-cancer-wisconsin-nomissing.data']);
X = data(:,2:10);
% X = zscore(X);
X = svmscale(X,[-1,1],'range','s');
y = data(:,11);
y(y==2) = 1;
y(y==4) = -1;
N = length(y);

perm = randperm(N);
index1 = perm(1:350);
index2 = setdiff(1:N,index1);
X1 = X(index1,:);
X2 = X(index2,:);
y1 = y(index1);
y2 = y(index2);

opts.lambda = 0.001;
opts.ktype = 'rbf';
opts.args = [12];
opts.method = 'benetreg';
% model = skFLDA(y1,X1,opts);
model = rvmtrain(y1,X1,opts);
yP1 = kpredict(X1,model);
yP2 = kpredict(X2,model);

yP1 = sign(yP1);
yP2 = sign(yP2);

index11 = find(y1==1);
index12 = find(y1==-1);
index13 = find(yP1~=y1);
index21 = find(y2==1);
index22 = find(y2==-1);
index23 = find(yP2~=y2);


figure;

subplot(221);
hold on;
scatter(X1(index11,1), X1(index11,2), 'bx');
scatter(X1(index12,1), X1(index12,2), 'ro');
scatter(X1(index13,1), X1(index13,2), 100, 'ko');
legend('c1', 'c2');
title(['train error num = ' num2str(length(index13))]);

subplot(222);
hold on;
scatter(X2(index21,1), X2(index21,2), 'bx');
scatter(X2(index22,1), X2(index22,2), 'ro');
scatter(X2(index23,1), X2(index23,2), 100, 'ko');
legend('c1', 'c2');
title(['test error num = ' num2str(length(index23))]);

subplot(224);
hold on;
scatter(X1(index11,1), X1(index11,2), 'bx');
scatter(X1(index12,1), X1(index12,2), 'ro');
scatter(model.sv(:,1), model.sv(:,2), 100, 'ko');
legend('c1', 'c2');
title('support vectors');
