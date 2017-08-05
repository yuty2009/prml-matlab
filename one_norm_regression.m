% One Norm Regression Demo
% 
% simulate y = sin(2*pi*[0:0.01:1]) by
% y = w0 + w1*x + w2*x^2 + w3*x^3 + ...+ wn*x^n
% min{ sum|y - t| }
% 
clc
clear all;

order = 9;
lambda = 0.0001;

load train_data;
train_size = size(x, 2);


PHI = zeros(train_size, order+1);
for i=1:train_size
    for j=1:order+1
        PHI(i,j) = PHI(i,j) + x(i)^(j-1);
    end
end

f = [zeros(1,order+1) 1]';
A = cat(1, PHI, -PHI);
A = cat(2, A, -ones(size(A,1),1));
A = cat(1, A, -f');
b = cat(1, y', -y');
b = [b; 0];
Wml = linprog(f, A, b);
Wml = Wml(1:order+1);

t = 0.01:0.01:1;
p_t = zeros(1,length(t));
for k=1:length(t)
    for i=1:order+1
        p_t(k) = p_t(k) + Wml(i)*t(k)^(i-1);
    end
end

load test_data;
p_y = zeros(1,train_size);
for k=1:train_size
    for i=1:order+1
        p_y(k) = p_y(k) + Wml(i)*tx(k)^(i-1);
    end
end

plot(t, sin(2*pi*t), '-r');hold on;
plot(t, p_t, '-b');hold on;
plot(x, y, 'ob'); hold on;
plot(tx, ty, 'og');
legend('standard sin(x)', 'predicted curve', 'trainset', 'testset');