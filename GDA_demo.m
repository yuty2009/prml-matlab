clc
clear

M = 1;
N1 = 1000;
N2 = 1500;
mu01 = 5;
mu02 = 0;
sigma0 = 3;

X01 = mvnrnd(mu01,sigma0,N1);
X02 = mvnrnd(mu02,sigma0,N2);
X0 = cat(1,X01,X02);
y01 = ones(N1,1);
y02 = zeros(N2,1);
y0 = cat(1,y01,y02);

[y11,X11] = hist(X01,100);
[y12,X12] = hist(X02,100);
y11 = y11/norm(y11);
y12 = y12/norm(y12);

[phi,mu21,mu22,sigma2] = GDA(y0,X0);
t1 = 0:0.1:10;
t2 = -5:0.1:5;
yt1 = normpdf(t1,mu21,sigma2);
yt2 = normpdf(t2,mu22,sigma2);

w0 = sigma0^(-1)*(mu01-mu02);
b0 = -(1/2)*mu01'*sigma0^(-1)*mu01 ...
    +(1/2)*mu02'*sigma0^(-1)*mu02 + log(phi/(1-phi));
w1 = sigma2^(-1)*(mu21-mu22);
b1 = -(1/2)*mu21'*sigma2^(-1)*mu21 ...
    +(1/2)*mu22'*sigma2^(-1)*mu22 + log(phi/(1-phi));
t3 = -10:0.1:15;
yt3 = 1./(1+exp(-(w0.*t3+b0)));
yt4 = 1./(1+exp(-(w1.*t3+b1)));
yt5 = phi*normpdf(t3,mu21,sigma2)./ ...
    (phi*normpdf(t3,mu21,sigma2)+(1-phi)*normpdf(t3,mu22,sigma2));

figure;
hold on;
plot(X11,y11,'r');
plot(X12,y12,'b');
plot(t1,yt1,'r--');
plot(t2,yt2,'b--');
plot(t3,yt3,'k');
plot(t3,yt4,'k--');
plot(t3,yt5,'g');
legend('data C1','data C2','hist C1','hist C2', ...
'sigmoid1','sigmoid2','posterior','Location','SouthEastOutside');