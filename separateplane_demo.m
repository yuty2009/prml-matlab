clc
clear

% load('data.mat');
load('tulip.mat');

index1 = find(y==1);
index2 = find(y==2);

X = X([1 end], :);
y(index2) = -1;
y = reshape(y, length(y), 1);

[W_lda B_lda] = LDA(y,X');

svmoption = ['-s 0 -t 0 -c 1 -g 0.001'];
model = svmtrain(y,X',svmoption);
W_svm = model.SVs' * model.sv_coef;
B_svm = -model.rho;

t = min(X(1,:)):0.1:max(X(1,:));
v1 = (-W_lda(1)*t - B_lda)/W_lda(2);
v2 = (-W_svm(1)*t - B_svm)/W_svm(2);

figure;
scatter(X(1,index1), X(2,index1), 'rx');
hold on;
scatter(X(1,index2), X(2,index2), 'bo');
hold on;
plot(t, v1, 'g-');
hold on;
plot(t, v2, 'r-');
legend('c1', 'c2', 'LDA', 'SVM');