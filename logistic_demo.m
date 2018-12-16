clc
clear

N1 = 200;
N2 = 200;
MU1 = [1.0;5.0];
MU2 = [2.5;2.5];
SIGMA1 = [0.3 0; 0 0.3];
SIGMA2 = [0.3 0; 0 0.3];

X1 = mvnrnd(MU1, SIGMA1, N1);
X2 = mvnrnd(MU2, SIGMA2, N2);

%% add some outliers
N3 = 50;
MU3 = [5.0;1.0];
SIGMA3 = [0.2 0; 0 0.2];
X3 = mvnrnd(MU3, SIGMA3, N3);

X12 = cat(1, X1, X2);
y12 = [ones(N1,1); -ones(N2,1)];

X123 = cat(1, X1, X2, X3);
y123 = [ones(N1,1); -ones(N2+N3,1)];

%% train with two clusters
[W_lda1, b_lda1] = FLDA(y12,X12);
[W_log1, b_log1] = logistic1(y12,X12);
svmoption = ['-s 0 -t 0 -c 1 -g 0.001'];
model1 = svmtrain(y12, X12, svmoption);
W_svm1 = model1.SVs' * model1.sv_coef;
b_svm1 = -model1.rho;

%% train with three clusters
[W_lda2, b_lda2] = FLDA(y123,X123);
[W_log2, b_log2] = logistic1(y123,X123);
model2 = svmtrain(y123, X123, svmoption);
W_svm2 = model2.SVs' * model2.sv_coef;
b_svm2 = -model2.rho;

%% calculate the separate line
t = [min(X123(:,1)):0.1:max(X123(:,1))];
v1 = (-W_lda1(1)*t - b_lda1)/W_lda1(2);
v2 = (-W_log1(1)*t - b_log1)/W_log1(2);
v3 = (-W_svm1(1)*t - b_svm1)/W_svm1(2);
v4 = (-W_lda2(1)*t - b_lda2)/W_lda2(2);
v5 = (-W_log2(1)*t - b_log2)/W_log2(2);
v6 = (-W_svm2(1)*t - b_svm2)/W_svm2(2);

%% visualize the results
figure;
subplot(121);
hold on;
scatter(X1(:,1), X1(:,2), 'bx');
scatter(X2(:,1), X2(:,2), 'ro');
plot(t, v1, 'g-');
plot(t, v2, 'r-');
plot(t, v3, 'b-');
axis([-1 6 0 8]);
legend('c1', 'c2', 'LDA', 'Logistic', 'SVM');
subplot(122);
hold on;
scatter(X1(:,1), X1(:,2), 'bx');
scatter(X2(:,1), X2(:,2), 'ro');
scatter(X3(:,1), X3(:,2), 'go');
plot(t, v4, 'g-');
plot(t, v5, 'r-');
plot(t, v6, 'b-');
axis([-1 6 0 8]);
legend('c1', 'c2', 'outliers', 'LDA', 'Logistic', 'SVM');
