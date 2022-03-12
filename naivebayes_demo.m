clc
clear

load wbcds.mat
% get labels and feature
y = wbcd(:, 2);
X = wbcd(:, 3:end);
% split the dataset into trainset and testset
N = length(y);
test_ratio = 0.2;
N_test = ceil(N*test_ratio);
N_train = N - N_test;
idx_total = randperm(N);
idx_train = idx_total(1:N_train);
idx_test = idx_total(N_train+1:end);
X_train = X(idx_train, :);
X_test = X(idx_test, :);
y_train = y(idx_train);
y_test = y(idx_test);

model = naivebayes(y_train, X_train, 'gauss');
y_pred = naivebayespredict(X_test, model);

accu = sum(y_pred == y_test)/length(y_test);
disp(['accu is ', num2str(accu)]);