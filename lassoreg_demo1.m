 
clc
clear

%% generate training samples y = sin(x)/x
N = 50;
sigma = 0.1;
X1 = linspace(-10, 10, N)';
Xt = linspace(-10, 10, N*10)';
y1 = sinc(X1/pi) + sigma*randn(N,1);

%% fixed basis feature extraction
order = 99;
% divided by 10 to avoid numerical problems
PHI1 = basispoly(X1/10,order); 
PHIt = basispoly(Xt/10,order);
% normlization of the features is very important
PHIa = cat(1,PHI1,PHIt);
% PHIa = zscore(PHIa);
PHIa = svmscale(PHIa,[-1,1],'range','s');
PHI1 = PHIa(1:N,:);
PHIt = PHIa(N+1:end,:);

%% train model
lambda = 1e-2;
%% L2 norm regularization
% [w0,b0] = ridgereg(y1, PHI1, lambda);
[w0,b0] = lassoreg(y1, PHI1, lambda);
%% L1 norm regularization
% [w1,b1] = lassoreg2(y1, PHI1, lambda);
%% mixed L1 and L2 norm regularization
% [w1,b1] = elasticnet(y1, PHI1, lambda, 1e-3);
%% Bayes L1 norm regularization
% [w0,b0] = blassoreg(y1, PHI1);
% [w1,b1] = blassoreg2(y1, PHI1);
%% Bayes ARD regularization
[w1,b1] = bardreg(y1, PHI1);
%% Variational Bayes linear regression
% [w1,b1] = vbayesreg(y1, PHI1);
%% Bayes elastic net regularization
% [w1,b1] = benetreg(y1, PHI1);

%% prediction
pt0 = PHIt*w0 + b0;
pt1 = PHIt*w1 + b1;

%% visualization
figure;
subplot(221);
hold on;
plot(Xt, sinc(Xt/pi), '-r');
plot(X1, y1, 'ob');
plot(Xt, pt0, '-b');
axis([-10 10 -0.4 1.2]);
legend('sin(x)/x','data','fitted');
subplot(222);
hold on;
plot(Xt, sinc(Xt/pi), '-r');
plot(X1, y1, 'ob');
plot(Xt, pt1, '-b');
axis([-10 10 -0.4 1.2]);
legend('sin(x)/x','data','fitted');
subplot(223);
bar(w0);
xlim([1,order+1]);
subplot(224);
bar(w1);
xlim([1,order+1]);
