%% Overfitting and Log evidence demo (Page 5-8 of PRML)

clc
clear all

N = 10;
sigma = 0.3;
x1 = linspace(0, 1, N)';
x2 = linspace(0, 1, N)';
t = linspace(0, 1, N*10)';
y1 = sin(2*pi*x1) + sigma*randn(N,1);
y2 = sin(2*pi*x2) + sigma*randn(N,1);

orders = 0:9;
K = length(orders);
rmse1 = zeros(K,1);
rmse2 = zeros(K,1);
yp1 = zeros(N, K);
yp2 = zeros(N, K);
yt = zeros(N*10, K);
evidence = zeros(K,1);
for i = 1:K
    order = orders(i);
    PHIt = basis(t, order);
    PHI1 = basis(x1, order);
    PHI2 = basis(x2, order); 
    
    lambda = 0;
	w = inv(PHI1'*PHI1 + lambda*eye(size(PHI1,2)))*PHI1'*y1;
    
    yt(:, i) = PHIt*w;
    yp1(:, i) = PHI1*w;
    yp2(:, i) = PHI2*w;
    rmse1(i) = sqrt(sum((yp1(:,i) - y1).^2)/N);
    rmse2(i) = sqrt(sum((yp2(:,i) - y2).^2)/N);
    
    %% Page 165-167 of PRML
    beta = 10;
    alpha = 5e-3;
    A = alpha*eye(size(PHI1,2)) + beta*PHI1'*PHI1;
    m = beta*A^(-1)*PHI1'*y1;
    evidence(i) = 0.5* (order*log(alpha) + N*log(beta) ...
        - beta*sum((yp1(:,i) - y1).^2) + alpha*m'*m ...
        - log(det(A)) - N*log(2*pi) );
end

figure;
plotindexs = [1 2 4 10];
for i = 1:length(plotindexs)
    subplot(3,2,i);
    hold on;
    scatter(x1, y1, 'o');
    scatter(x2, y2, '*');
    ezplot('sin(2*pi*x)', [0 1 -2 2]);
    plot(t, yt(:,plotindexs(i)), '-r');
    title(['order = ' num2str(orders(plotindexs(i)))]);
    % legend('samples', 'sin(2*pi*x)', 'fitted curve');
end
subplot(3,2,i+1);
hold on;
plot(orders, rmse1, '-bo');
plot(orders, rmse2, '-ro');
title('root-mean-square error vs. order');
% legend('Training', 'test', 'Location', 'NW');
subplot(3,2,i+2);
plot(orders, evidence, '-bo');
title('log evidence vs. order');