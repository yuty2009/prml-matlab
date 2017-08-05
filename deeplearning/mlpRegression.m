clc
clear

N1 = 10;
x1 = linspace(0,1,N1)';
sigma = 0.3;
y1 = sin(2*pi*x1) + sigma*randn(N1,1);

Nt = 100;
xt = linspace(0,1,Nt)';

NHs = [1 2 3 10];
figure;
for i = 1:length(NHs)
    %% train model
    opts.verbose = 'on';
    opts.maxepochs = 400;
    opts.optmethod = 'lbfgs';
    mlp = mlpinit([1,NHs(i),1]);
    mlp.oTF = 'linear';
    mlp = mlptrain(mlp,y1,x1,opts);

    %% predict
    yt = mlppredict(mlp,xt);

    %% visualize
    subplot(2,length(NHs)/2,i);
    hold on;
    plot(x1, y1, 'ob');
    plot(xt, yt, '-r');
    legend('trainset', 'fitted');
    title(['M = ' num2str(NHs(i))]);
end