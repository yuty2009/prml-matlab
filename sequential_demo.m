%% Sequential learning demo
clc
clear

t = [0.01:0.01:1]';
N = length(t);
order = 6;

%% generate training samples
idx1 = randi(N, [1, N]);
x1 = t(idx1);
sigma = 0.1; % standard deviation of noise
y1 = sin(2*pi*x1) + sigma*randn(N,1);
PHI1 = basis(x1, order);

%% train model
eta = 0.2;
% [w, b] = lsseq(y1, PHI1, eta);
[w, b] = bayesseq(y1, PHI1);

PHIt = basis(t, order);
pt = PHIt*w + b;

%% visualize
figure;
hold on;
plot(t, sin(2*pi*t), '-r');
plot(t, pt, '-b');
plot(x1, y1, 'ob');
legend('standard sin(x)', 'predicted curve', 'trainset');