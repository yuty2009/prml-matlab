clc
clear

N = 50;
sigma = 0.1;
X1 = linspace(-10, 10, N)';
Xt = linspace(-10, 10, N*10)';
y1 = sinc(X1/pi) + sigma*randn(N,1);

opts.ktype = 'rbf';
opts.args = [1.2];
opts.lambda = 1e-4;
opts.method = 'blassoreg';
model1 = kridge(y1,X1,opts);
% model2 = skridge(y1,X1,opts);
model2 = rvmtrain(y1,X1,opts);
% model2 = gpreg(y1,X1,opts);
yt1 = kpredict(Xt,model1);
% yt2 = kpredict(Xt,model2);
yt2 = kpredict(Xt,model2);
% yt2 = kpredict(Xt,model2);

%% visualize
figure;
hold on;
scatter(X1, y1, 'o');
scatter(X1(model2.svind), y1(model2.svind), 100, 'or');
ezplot('sinc(x/pi)', [-10 10 -0.4 1.2]);
plot(Xt, yt1, '-r');
plot(Xt, yt2, '-g');
legend('trainset','sv','sin(2*pi*x)','kridge','rvm');
