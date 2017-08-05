clc
clear

t = 0.1:0.1:10;
x1 = sin(2*pi*t);
x2 = sawtooth(pi*t);
X = [x1;x2];

A0 = randn(2);
Y = A0*X;

% [W,sphere,compvars,bias,signs,lrates,icasig] = runica(Y);
% [icasig,A,W] = fastica(Y);
[icasig,A,W] = ICAML(Y);

figure;
subplot(321);
plot(t,X(1,:));
subplot(322);
plot(t,X(2,:));

subplot(323);
plot(t,Y(1,:));
subplot(324);
plot(t,Y(2,:));

subplot(325);
plot(t,icasig(1,:));
subplot(326);
plot(t,icasig(2,:));