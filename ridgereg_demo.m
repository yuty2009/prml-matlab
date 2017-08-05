%% Least Square Regression with norm2 regularization
% 
% simulate y = sin(2*pi*[0:0.01:1]) by
% y = w0 + w1*x + w2*x^2 + w3*x^3 + ...+ wn*x^n
% min{ |y - t|^2 + lambda*w'*w }
% 
clc
clear

sigma = 0.3; % standard deviation of noise
t = [0.01:0.01:1]';
N = length(t);
order = 9;

%% generate training samples
idx1 = randi(N, [1, N]);
x1 = t(idx1);
y1 = sin(2*pi*x1) + sigma*randn(N,1);
PHI1 = basis(x1, order);

%% train model
lambda = 0.01;
[w, b] = ridgereg(y1, PHI1, lambda);

%% generate testing samples
idx2 = randi(N, [1, N]);
x2 = t(idx2);
y2 = sin(2*pi*x2) + sigma*randn(N,1);
PHI2 = basis(x2, order);

%% predict
py = PHI2*w + b;
% py = kridgereg(y1, PHI1, PHI2, lambda, 'poly', 5, 0);

PHIt = basis(t, order);
pt = PHIt*w + b;
% pt = kridgereg(y1, PHI1, PHIt, lambda, 'poly', 5, 0);

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