%% check crossvalidation resulted parameter
clc
clear

%% generate train dataset
% designed signal
N1 = 128; % number of samples in class 1
N2 = 128; % number of samples in class 2
N = N1 + N2;
P = 1024; % feature dimension
X1 = randn(N1,P);
X2 = randn(N2,P) - 10;
XRaw = cat(1, X1, X2);
% designed weights
NG = 32; % number of groups
PG = floor(P/NG); % number of feature per-group
groups = ceil([1:P]/PG);
NSG = 10; % number of active groups
perm = randperm(NG);
actives = perm(1:NSG);
w0 = zeros(P,1);
for i = 1:NSG
%     w0(find(groups==actives(i))) = ...
%       randn(size(w0(find(groups==actives(i))))); % gaussian signal
    w0(find(groups==actives(i))) = ...
        ones(size(w0(find(groups==actives(i))))); % uniform signal
end
% design matrix and class label
sigma = 0.2;
X0 = XRaw*diag(w0) + sigma*randn(N,P);
y0 = [ones(N1,1); -1*ones(N2,1)];
% permutate the samples
perm = randperm(N);
X = X0(perm,:);
y = y0(perm,1);
% calculate the discriminability of each feature
rr = rsquare(y,X);

% save('classification_gaussian', 'X', 'y', 'NG', 'w0');
% load('classification_gaussian');

%% perform crossvalidation
methods = {'bardlog','bardgrouplog'};
% methods = {'logistic','loglasso','loggrouplasso','bayeslog','bardlog','bardgrouplog'};
numMethods = length(methods);

XTrain = X;
yTrain = y;

ws = zeros(size(X,2), numMethods);
for m = 1:numMethods
    method = methods{m};
    disp(method);
    switch(method)
        case 'flda'
            fisher = FLDA(yTrain, XTrain);
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
        case 'logistic'
            logmodel = logistic(yTrain, XTrain, 0.001);
            model = logmodel;
            ws(:,m) = model.b;
        case 'loglasso'
            loglasso = lassolog(yTrain, XTrain, 0.001);
            model = loglasso;
            ws(:,m) = model.b;
        case 'loggrouplasso'
            loggrouplasso = lassogrouplog(yTrain, XTrain, groups, 0.001);
            model = loggrouplasso;
            ws(:,m) = model.b;
        case 'lognuclear'
            lognuclear = nuclearlog(yTrain, XTrain, NG, 0.001);
            model = lognuclear;
            ws(:,m) = model.b;
        case 'bayeslog'
            bayeslogmodel = bayeslog(yTrain, XTrain);
            model = bayeslogmodel;
            ws(:,m) = model.b;
        case 'bardlog'
            bardlogmodel = bardlog(yTrain, XTrain);
            model = bardlogmodel;
            ws(:,m) = model.b;
        case 'bardgrouplog'
            bardloggroup = bardgrouplog(yTrain, XTrain, groups);
            model = bardloggroup;
            ws(:,m) = model.b;
        otherwise
            disp('unknown method');
    end
end

%% visualize
figure;
subplot(numMethods+2,1,1);
plot(w0);
axis tight;
ylim([-2 2]);
title('raw');
subplot(numMethods+2,1,2);
plot(rr);
axis tight;
ylim([0 1]);
title('r^2');
for m = 1:numMethods
    subplot(numMethods+2,1,m+2);
    plot(ws(:,m));
    axis tight;
    % ylim([-2 2]);
    title(methods{m});
end