clc
clear

M = 2;

% generate train dataset
N1 = 20; % number of samples in class 1
N2 = 30; % number of samples in class 2
X1 = rand(N1,M);
X2 = rand(N2,M) + [ones(N2,1) 0.5*ones(N2,1)];

X = cat(1, X1, X2);
y = [ones(N1,1); -1*ones(N2,1)];

[W_lda, b_lda] = FLDA(y, X);

svmoption = ['-s 0 -t 0 -c 1 -g 0.001'];
svmmodel = svmtrain(y, X, svmoption);
W_svm = svmmodel.SVs' * svmmodel.sv_coef;
b_svm = -svmmodel.rho;

y1 = sign(X1*W_lda + b_lda);
y2 = sign(X2*W_lda + b_lda);

% generate test dataset
N11 = 50;
N21 = 50;
X11 = rand(N11,M);
X21 = rand(N21,M) + [ones(N21,1) 0.5*ones(N21,1)];
XT = cat(1, X11, X21);
yT = [ones(N11,1); -1*ones(N21,1)];
% randomize order of test samples
indices = randperm(size(XT,1));
XT = XT(indices,:);
yT = yT(indices);

yP = sign(XT*W_lda + b_lda);
index1 = find(yT == 1);
index2 = find(yT == -1);
index3 = find(yT ~= yP);

t = [min(X(:,1)):0.1:max(X(:,1))];
v1 = (-W_lda(1)*t - b_lda)/W_lda(2);
v2 = (-W_svm(1)*t - b_svm)/W_svm(2);

figure;

subplot(131);
hold on;
scatter(X1(:,1), X1(:,2), 'bx');
scatter(X2(:,1), X2(:,2), 'ro');
plot(t, v1, 'g-');
plot(t, v2, 'r-');
axis([0 2 -2 4]);
legend('c1', 'c2', 'LDA', 'SVM');

subplot(132);
hold on;
scatter(X11(:,1), X11(:,2), 'bx');
scatter(X21(:,1), X21(:,2), 'ro');
plot(t, v1, 'g-');
plot(t, v2, 'r-');
axis([0 2 -2 4]);
legend('c1', 'c2', 'LDA', 'SVM');

subplot(133);
hold on;
scatter(XT(index1,1), XT(index1,2), 'bx');
scatter(XT(index2,1), XT(index2,2), 'ro');
scatter(XT(index3,1), XT(index3,2), 100, 'ko');
axis([0 2 -2 4]);
legend('c1', 'c2');
