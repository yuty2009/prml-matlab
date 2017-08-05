clc
clear

N = 10;
sigma = 0.3;
x1 = linspace(0, 1, N)';
t = linspace(0, 1, N*10)';
y1 = sin(2*pi*x1) + sigma*randn(N,1);

order = 9;
PHI1 = basis(x1, order);
PHIt = basis(t, order);

lambda = 0.01;
qs = [1 2 4 Inf];
Nq = length(qs);
ws = zeros(Nq,order+1);
for i = 1:Nq
    q = qs(i);
    
    cvx_begin
        variable w(order+1,1);
        minimize norm(PHI1*w-y1,2) + lambda*norm(w,q)
    cvx_end
    ws(i,:) = w;
end

figure;
for i = 1:Nq
    w = ws(i,:)';
    yt = PHIt*w;
    
    subplot(2,Nq,i);
    hold on;
    scatter(x1, y1, 'o');
    ezplot('sin(2*pi*x)', [0 1 -2 2]);
    plot(t, yt, '-r');
    
    subplot(2,Nq,Nq+i);
    stem(w);
    title(['q = ' num2str(qs(i))]);
end