%% Overfitting demo (Page 5-8 of PRML)

clc
clear

N = 10;
sigma = 0.3;
x1 = linspace(0, 1, N)';
t = linspace(0, 1, N*10)';
y1 = sin(2*pi*x1) + sigma*randn(N,1);

order = 9;
lambdas = [1e-20, 1e-5, 1, 1e5];
K = length(lambdas);
yt = zeros(N*10, K);
for i = 1:K
    PHIt = basis(t, order);
    PHI1 = basis(x1, order);
    
    lambda = lambdas(i);
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
    title(['log_{10}(\lambda) = ' num2str(log10(lambdas((i))))]);
end