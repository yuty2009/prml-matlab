%% Overfitting demo (Page 5-8 of PRML)

clc
clear

order = 9;
Ns = [10 20 50 100];
K = length(Ns);
t = linspace(0, 1, 1000)';
yt = zeros(1000, K);
figure;
for i = 1:K
    sigma = 0.3;
    N = Ns(i);
    x1 = linspace(0, 1, N)';
    y1 = sin(2*pi*x1) + sigma*randn(N,1);
    
    PHIt = basis(t, order);
    PHI1 = basis(x1, order);
    
    lambda = 0;
	w = inv(PHI1'*PHI1 + lambda*eye(size(PHI1,2)))*PHI1'*y1;
    yt(:, i) = PHIt*w;
    
    subplot(2,2,i);
    hold on;
    scatter(x1, y1, 'o');
    ezplot('sin(2*pi*x)', [0 1 -2 2]);
    plot(t, yt(:,i), '-r');
    title(['N = ' num2str(Ns(i))]);
end
