
clc
clear

%% bivariate example
% f(x,y) = C_x^n y^{x+a-1}(1-y)^{n-x+b-1}
% x = 0,1,...,n   0<=y<=1
% f(x|y) is Binomial(n,y)
% f(y|x) is Beta(x+a,n-x+b)

n = 16;
a = 2;
b = 4;
y0 = 0.5;

r = 2;
m = 500;
x = zeros(m,r);
for i = 1:r
    for j = 1:m
        x1 = random('bino',n,y0);
        x(j,i) = x1;
        y1 = random('beta',x1+a,n-x1+b);
        y0 = y1;
    end
end

figure;
hist(x,16);


% using the final values from Gibbs sequences of length k
k = 10;
x = zeros(m,1);
y = zeros(m,1);
for i = 1:m
    for j = 1:k
        x1 = random('bino',n,y0);
        y1 = random('beta',x1+a,n-x1+b);
        y0 = y1;
    end
    x(i) = x1;
    y(i) = y1;
end

[nx,xout] = hist(x,16);

t = min(x):max(x);
ft = zeros(length(t),1);
for i = 1:length(t)
    ft(i) = mean(pdf('bino',t(i),n,y));
end

figure;
hold on;
bar(xout,nx/m);
plot(t,ft);
