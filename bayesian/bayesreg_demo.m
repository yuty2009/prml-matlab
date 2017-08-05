% bayesian linear regression

clc
clear all

order = 9;

sigma = 0.3; % standard deviation of noise
t = [0.01:0.01:1]';
N = length(t);

%% generate training samples
idx1 = randi(N, [1, N]);
x1 = t(idx1);
y1 = sin(2*pi*x1) + sigma*randn(N,1);
PHI1 = basis(x1, order);

%% train model
% [w, b] = bayesard(y1, PHI1);
[w, b] = bayesard_grouped(y1, PHI1, 100);

%% generate testing samples
idx2 = randi(N, [1, N]);
x2 = t(idx2);
y2 = sin(2*pi*x2) + sigma*randn(N,1);
PHI2 = basis(x2, order);

%% predict
py = PHI2*w + b;

PHIt = basis(t, order);
pt = PHIt*w + b;

%% visualize
figure;
subplot(121);
hold on;
plot(t, sin(2*pi*t), '-r');
plot(t, pt, '-b');
plot(x1, y1, 'ob');
plot(x2, y2, 'og');
legend('standard sin(x)', 'predicted curve', 'trainset', 'testset');
subplot(122);
scatter(py, y2-py);
xlabel('fitted');
ylabel('residual');