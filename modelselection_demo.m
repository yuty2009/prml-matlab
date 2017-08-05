%% Overfitting and regularization demo (Page 5-8 of PRML)

clc
clear all

N = 10;
sigma = 0.3;
x1 = linspace(0, 1, N)';
x2 = linspace(0, 1, N)';
t = linspace(0, 1, N*10)';
y1 = sin(2*pi*x1) + sigma*randn(N,1);
y2 = sin(2*pi*x2) + sigma*randn(N,1);

order = 9;
lambdas = logspace(-20, 2, 40);
loglambdas = log(lambdas);
K = length(lambdas);
rmse1 = zeros(K,1);
rmse2 = zeros(K,1);
yp1 = zeros(N, K);
yp2 = zeros(N, K);
yt = zeros(N*10, K);
evidence = zeros(K,1);
for i = 1:K
    PHIt = basispoly(t, order);
    PHI1 = basispoly(x1, order);
    PHI2 = basispoly(x2, order); 
    
    lambda = lambdas(i);
	w = inv(PHI1'*PHI1 + lambda*eye(size(PHI1,2)))*PHI1'*y1;
    
    yt(:, i) = PHIt*w;
    yp1(:, i) = PHI1*w;
    yp2(:, i) = PHI2*w;
    rmse1(i) = sqrt(sum((yp1(:,i) - y1).^2)/N);
    rmse2(i) = sqrt(sum((yp2(:,i) - y2).^2)/N);
    
    %% Page 165-167 of PRML
    beta = 10;
    alpha = beta*lambda; % note this
    A = alpha*eye(size(PHI1,2)) + beta*PHI1'*PHI1;
    m = beta*A^(-1)*PHI1'*y1;
    evidence(i) = 0.5* (order*log(alpha) + N*log(beta) ...
        - beta*sum((yp1(:,i) - y1).^2) + alpha*m'*m ...
        - log(det(A)) - N*log(2*pi) );
end

figure;
plotindexs = [1 10 25 40];
for i = 1:length(plotindexs)
    subplot(3,2,i);
    hold on;
    scatter(x1, y1, 'o');
    scatter(x2, y2, '*');
    ezplot('sin(2*pi*x)', [0 1 -2 2]);
    plot(t, yt(:,plotindexs(i)), '-r');
    title(['log(\lambda) = ' num2str(loglambdas(plotindexs(i)))]);
    % legend('samples', 'sin(2*pi*x)', 'fitted curve');
end
subplot(3,2,i+1);
hold on;
plot(loglambdas, rmse1, '-b');
plot(loglambdas, rmse2, '-r');
title('root-mean-square error vs. log(\lambda)');
% legend('Training', 'test', 'Location', 'NW');
subplot(3,2,i+2);
plot(loglambdas, evidence, '-b');
title('log evidence vs. log(\lambda)');