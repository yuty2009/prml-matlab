clc
clear

figure;

N0 = 1000;
x0 = rand(N0,1);

subplot(121);
hist(x0,10);

K = 10000;
x = zeros(N0,1);
for i = 1:K
    x1 = rand(N0,1);
    x = x + x1;
end
x = x/K;

subplot(122);
hist(x,100);