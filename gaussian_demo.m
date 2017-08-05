clc
clear

figure;

%% 1-D disttribution demo
mu = 0;
sigma = 1;
x = -5:0.1:5;
N = length(x);
for i = 1:N
    y(i) = (2*pi*sigma)^(-1/2)*exp(-(1/(2*sigma))*(x(i)-mu)^2);
end

subplot(231);
plot(x,y,'k');
title('1-D Gaussian');
grid on;box off;

hold on;

mu = 0;
sigma = 4;
x = -5:0.1:5;
N = length(x);
for i = 1:N
    y(i) = (2*pi*sigma)^(-1/2)*exp(-(1/(2*sigma))*(x(i)-mu)^2);
end

plot(x,y,'m');
legend(['sigma = ' num2str(1)],['sigma = ' num2str(sigma)]);

%% 2-D disttribution demo
mu = [0 0];
sigma = [1 0;0 1];
x1 = -5:0.1:5;
x2 = -5:0.1:5;
N1 = length(x1);
N2 = length(x2);
for i = 1:N1
    for j = 1:N2
        y(i,j) = (2*pi)^(-2/2)*det(sigma)^(-1/2) ...
            *exp(-(1/2)*([x(i),x(j)]-[mu(1),mu(2)])*sigma^(-1) ...
            *([x(i),x(j)]-[mu(1),mu(2)])');
    end
end

subplot(232);
mesh(x1,x2,y);
title('2-D Gaussian');
grid on;box off;
zlim([0 0.2]);
view(3);

mu = [0 0];
sigma = [2 0;0 2];
x1 = -5:0.1:5;
x2 = -5:0.1:5;
N1 = length(x1);
N2 = length(x2);
for i = 1:N1
    for j = 1:N2
        y(i,j) = (2*pi)^(-2/2)*det(sigma)^(-1/2) ...
            *exp(-(1/2)*([x(i),x(j)]-[mu(1),mu(2)])*sigma^(-1) ...
            *([x(i),x(j)]-[mu(1),mu(2)])');
    end
end

subplot(233);
mesh(x1,x2,y);
title('2-D Gaussian');
grid on;box off;
zlim([0 0.2]);
view(3);

%% 2-D Gaussian distribution with different covariance matrix forms
mu1 = [0 0];
mu2 = [0 0];
mu3 = [0 0];
sigma1 = [1 0;0 1];
sigma2 = [1 0;0 2];
sigma3 = [1 0.8;0.8 2];

x1 = -5:0.1:5;
x2 = -5:0.1:5;
N1 = length(x1);
N2 = length(x2);
for i = 1:N1
    for j = 1:N2
        y1(i,j) = (2*pi)^(-2/2)*det(sigma1)^(-1/2) ...
            *exp(-(1/2)*([x1(i),x2(j)]-[mu1(1),mu1(2)])*sigma1^(-1) ...
            *([x1(i),x2(j)]-[mu1(1),mu1(2)])');
        y2(i,j) = (2*pi)^(-2/2)*det(sigma2)^(-1/2) ...
            *exp(-(1/2)*([x1(i),x2(j)]-[mu2(1),mu2(2)])*sigma2^(-1) ...
            *([x1(i),x2(j)]-[mu2(1),mu2(2)])');
        y3(i,j) = (2*pi)^(-2/2)*det(sigma3)^(-1/2) ...
            *exp(-(1/2)*([x1(i),x2(j)]-[mu3(1),mu3(2)])*sigma3^(-1) ...
            *([x1(i),x2(j)]-[mu3(1),mu3(2)])');
        % y1(i,j) = mvnpdf([x1(i) x2(j)],mu1,sigma1);
        % y2(i,j) = mvnpdf([x1(i) x2(j)],mu2,sigma2);
        % y3(i,j) = mvnpdf([x1(i) x2(j)],mu3,sigma3);
    end
end

subplot(234);
contour(x1,x2,y1,3);
title('isotropic');
grid on;box off;
axis equal;
subplot(235);
contour(x1,x2,y2,3);
grid on;box off;
axis equal;
title('diagonal');
subplot(236);
contour(x1,x2,y3,3);
grid on;box off;
axis equal;
title('general');