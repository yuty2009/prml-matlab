clc
clear

N1 = 200;
N2 = 200;
N3 = 50;
MU1 = [3.0;3.0];
MU2 = [4.0;5.0];
MU3 = [3.0;1.0];
SIGMA1 = [2.0 -0.2; -0.2 1.0];
SIGMA2 = [1.0 0; 0 1.0];
SIGMA3 = [1.0 0.2; 0.2 1.0];
X1 = mvnrnd(MU1, SIGMA1, N1);
X2 = mvnrnd(MU2, SIGMA2, N2);
X3 = mvnrnd(MU3, SIGMA3, N3);

X = cat(1, X1, X2, X3);
y = [ones(N1,1); 2*ones(N2+N3,1)];

[N,P] = size(X);

SN = [P,6,length(unique(y))];

opts.verbose = 'on';
opts.maxepochs = 400;
% opts.method = 'minibatch';
% opts.batchsize = 10;
% opts.lrate = 0.1;
opts.optmethod = 'lbfgs';
mlp = mlpinit(SN);
mlp.oTF = 'softmax';
mlp = mlptrain(mlp,y,X,opts);

xmin1 = min(X(:,1));
xmin2 = min(X(:,2));
xmax1 = max(X(:,1));
xmax2 = max(X(:,2));
x1 = xmin1:0.2:xmax1;
x2 = xmin2:0.2:xmax2;
for i = 1:length(x1)
    for j = 1:length(x2)
        Xt((i-1)*length(x2)+j,:) = [x1(i),x2(j)];
    end
end

yt = mlppredict(mlp,Xt);

idx11 = find(y==1);
idx12 = find(y==2);
idx21 = find(yt==1);
idx22 = find(yt==2);

figure;
subplot(121);
hold on;
scatter(X1(:,1),X1(:,2),'ro');
scatter(X2(:,1),X2(:,2),'bx');
scatter(X3(:,1),X3(:,2),'gx');
subplot(122);
hold on;
scatter(Xt(idx21,1),Xt(idx21,2),'ro');
scatter(Xt(idx22,1),Xt(idx22,2),'bx');