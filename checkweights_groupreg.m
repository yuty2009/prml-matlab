%% check crossvalidation resulted parameter
clc
clear

%% generate train dataset
% design matrix
N = 256;
P = 1024; % feature dimension
X = randn(N,P);
% designed weights
NG = 32; % number of groups
PG = floor(P/NG); % number of feature per-group
groups = ceil([1:P]/PG);
NSG = 10; % number of active groups
perm = randperm(NG);
actives = perm(1:NSG);
w0 = zeros(P,1);
for i = 1:NSG
    w0(find(groups==actives(i))) = ...
      randn(size(w0(find(groups==actives(i))))); % gaussian signal
%   w0(find(groups==actives(i))) = ...
%       ones(size(w0(find(groups==actives(i))))); % uniform signal
end
% regression target
sigma = 0.2;
y = X*w0 + sigma*randn(N,1);

% save('regression_gaussian', 'X', 'y', 'NG', 'w0');
% load('regression_gaussian');

%% perform crossvalidation
methods = {'ridge','lasso','glasso','bard','bgard'};
numMethods = length(methods);

XTrain = X;
yTrain = y;

ws = zeros(size(X,2), numMethods);
for m = 1:numMethods
    method = methods{m};
    disp(method);
    switch(method)
        case 'flda'
            fisher = LDA(yTrain, XTrain);
            model = fisher;
            ws(:,m) = model.b;
        case 'swlda'
            [b,se,pval,inmodel,stats,nextstep,history] ...
                = stepwisefit(XTrain,yTrain,'penter',0.10,'premove',0.15, ...
                'scale','on','display','off');
            swlda.b = b;
            model = swlda;
            ws(:,m) = model.b;
        case 'svm'
            svmoption = ['-s 0 -t 0 -c 1 -g 0.001'];
            svmmodel = svmtrain(yTrain,XTrain,svmoption);
            model = svmmodel;
            ws(:,m) = model.SVs' * model.sv_coef;
        case 'ridge'
            ridge = ridgereg(yTrain, XTrain, 10^(-2));
            model = ridge;
            ws(:,m) = model.b;
        case 'lasso'
            lasso = lassoreg(yTrain, XTrain, 10^(-4));
            model = lasso;
            ws(:,m) = model.b;
        case 'glasso'
            lassogroup1 = glassoreg(yTrain, XTrain, NG, 10^(-5));
            model = lassogroup1;
            ws(:,m) = model.b;
        case 'enet'
            enet = elasticnet(yTrain, XTrain, 1e-4, 1e-3);
            model = enet;
            ws(:,m) = model.b;
        case 'bayes'
            bayes = bayesreg(yTrain, XTrain);
            model = bayes;
            ws(:,m) = model.b;
        case 'bard'
            bard = bardreg(yTrain, XTrain);
            model = bard;
            ws(:,m) = model.b;
        case 'bgard'
            bardgroup = bgardreg(yTrain, XTrain, NG);
            model = bardgroup;
            ws(:,m) = model.b;
        case 'vbgard'
            bvarsgroup = vbgardreg(yTrain, XTrain, NG);
            model = bvarsgroup;
            ws(:,m) = model.b;
        case 'bglasso'
            blassogroup = bglassoreg(yTrain, XTrain, NG);
            model = blassogroup;
            ws(:,m) = model.b;
        case 'benet'
            benet = benetreg(yTrain, XTrain);
            model = benet;
            ws(:,m) = model.b;
        otherwise
            disp('unknown method');
    end
end

%% visualize
figure;
subplot(numMethods+1,1,1);
plot(w0);
axis tight;
ylim([-2 2]);
title('raw');
for m = 1:numMethods
    subplot(numMethods+1,1,m+1);
    plot(ws(:,m));
    axis tight;
    ylim([-2 2]);
    title(methods{m});
end