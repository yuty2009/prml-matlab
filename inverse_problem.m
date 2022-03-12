clc
clear

N = 100;
ax = 0:0.01:1;
at = 0:0.01:1;
x = rand(1,N);
t = x + 0.3.*sin(2*pi*x) + ( -0.1 + 0.2.*rand(size(x)) );

net1 = newff(x,t,6);
net1.trainParam.epochs = 5;
net1 = train(net1,x,t);
y = sim(net1,ax);

net2 = newff(t,x,6);
net2.trainParam.epochs = 5;
net2 = train(net2,t,x);
x1 = sim(net2,at);

subplot(121);
scatter(x,t); hold on;
plot(ax,y);
axis([0 1 0 1]);

subplot(122);
scatter(t,x); hold on;
plot(at,x1);
axis([0 1 0 1]);