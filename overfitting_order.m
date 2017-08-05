%% Overfitting demo (Page 5-8 of PRML)

clc
clear

N = 10;
sigma = 0.3;
x1 = linspace(0, 1, N)';
t = linspace(0, 1, N*10)';
y1 = sin(2*pi*x1) + sigma*randn(N,1);

orders = [0 1 3 9];
K = length(orders);
yt = zeros(N*10, K);
for i = 1:K
    order = orders(i);
    PHIt = basis(t, order);
    PHI1 = basis(x1, order);
    
    lambda = 0;
	w = inv(PHI1'*PHI1 + lambda*eye(size(PHI1,2)))*PHI1'*y1;
    yt(:, i) = PHIt*w;
end

figure;
for i = 1:K
    subplot(2,2,i);
    hold on;
    scatter(x1, y1, 'o');
    ezplot('sin(2*pi*x)', [0 1 -2 2]);
    plot(t, yt(:,i), '-r');
    title(['M = ' num2str(orders(i))]);
end