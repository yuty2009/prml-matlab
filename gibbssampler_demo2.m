
clc
clear

%% bivariate example
% f(x|y) \propto y e^{-yx}, 0 < x < B < \infty
% f(y|x) \propto x e^{-xy}, 0 < y < B < \infty
% B is a known positive constant

B = 5;
k = 15;
m = 5000;
r = 1;
y0 = 1;
x = zeros(m,1);
y = zeros(m,1);
for i = 1:m
    for j = 1:k
        x1 = random('exp',1/y0);
        while (x1 <=0 || x1 >= B)
            x1 = random('exp',1/y1);
        end
        y1 = random('exp',1/x1);
        while (y1 <=0 || y1 >= B)
            y1 = random('exp',1/x1);
        end
        y0 = y1;
    end
    x(i) = x1;
    y(i) = y1;
end

[nx,xout] = hist(x,50);

t = 0.05:0.05:B-0.05;
ft = zeros(length(t),1);
for i = 1:length(t)
    ft(i) = mean(pdf('exp',t(i),1./y));
end

figure;
hold on;
bar(xout,nx/m);
plot(t,ft);
