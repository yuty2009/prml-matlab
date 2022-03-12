%% Least Square Regression with norm2 regularization
% 
% simulate y = sin(2*pi*[0:0.01:1]) by
% y = w0 + w1*x + w2*x^2 + w3*x^3 + ...+ wn*x^n
% min ||y - t||^2
%
clc
clear

N = 10;
sigma = 0.3;
x1 = linspace(0, 1, N)';
t = linspace(0, 1, N*10)';
y1 = sin(2*pi*x1) + sigma*randn(N,1);

order = 6;
PHIt = basis(t, order);
PHI1 = basis(x1, order);

%% learning model parameters
w = lsreg(y1, PHI1);

yt = PHIt*w;

%% visualize
figure;
hold on;
scatter(x1, y1, 'o');
ezplot('sin(2*pi*x)', [0 1 -2 2]);
plot(t, yt, '-r');