%% crossvalidation demo for classification
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
groups = ceil([1:P]'/PG);
NSG = 10; % number of active groups
perm = randperm(NG);
actives = perm(1:NSG);
w0 = zeros(P,1);
for i = 1:NSG
    w0(find(groups==actives(i))) = ...
      randn(size(w0(find(groups==actives(i))))); % gaussian signal
%     w0(find(groups==actives(i))) = ...
%         ones(size(w0(find(groups==actives(i))))); % uniform signal
end
% design matrix and class label
sigma = 0.2;
X0 = XRaw*diag(w0) + sigma*randn(N,P);
y0 = [ones(N1,1); -1*ones(N2,1)];
% permutate the samples
perm = randperm(N);
X = X0(perm,:);
y = y0(perm,1);

figure;
hold on;
scatter(X1(:,1), X1(:,2), 'bx');
scatter(X2(:,1), X2(:,2), 'ro');

%% perform crossvalidation
numFolds = 10;
numTotal = length(y);
numTest = floor(numTotal/numFolds);
numTrain = numTotal - numTest;
indexTotal = randperm(numTotal); % [1:numTotal]; %

methods = {'bayeslog','bardlog','bardloggroup'};
% methods = {'ridge','lasso','lassogroup1','bayes','bard','bardgroup','bayeslog','bardlog'};
numMethods = length(methods);

params = 1e-2;%logspace(-10,5,20);%
numParams = length(params);

disp([num2str(numFolds) ' fold cross-validation']);
accuracy = zeros(numFolds, numParams, numMethods);
for fold = 1:numFolds
    indexTest = indexTotal((fold-1)*numTest+1:fold*numTest);
    indexTrain = setdiff(indexTotal,indexTest);
    
    XTest = X(indexTest,:);
    XTrain = X(indexTrain,:);
    yTest = y(indexTest);
    yTrain = y(indexTrain);
    
    for p = 1:numParams
        lambda = params(p);

        for m = 1:numMethods
            method = methods{m};
        	disp(['fold = ' num2str(fold) ', lambda = ' num2str(lambda) ', method = ' method]);
            switch(method)
                case 'flda'
                    fisher = LDA(yTrain, XTrain);
                    model = fisher;
                case 'swlda'
                    [b,se,pval,inmodel,stats,nextstep,history] = stepwisefit(XTrain,yTrain,'penter',0.10,'premove',0.15,'scale','on','display','off');
                    swlda.b = b;
                    model = swlda;
                case 'svm'
                    svmoption = ['-s 0 -t 0 -c 1 -g 0.001'];
                    svmmodel = svmtrain(yTrain,XTrain,svmoption);
                    model = svmmodel;
                case 'ridge'
                    ridge = ridgereg(yTrain, XTrain, lambda);
                    model = ridge;
                case 'lasso'
                    lasso = lassoreg(yTrain, XTrain, lambda);
                    model = lasso;
                case 'lassogroup'
                    lassogroup = lassoreg_grouped1(yTrain, XTrain, NG, lambda);
                    model = lassogroup;
                case 'bayes'
                    bayes = bayesreg(yTrain, XTrain);
                    model = bayes;
                case 'bard'
                    bard = bayesard(yTrain, XTrain);
                    model = bard;
                case 'bardgroup'
                    bardgroup = bayesard_grouped(yTrain, XTrain, NG);
                    model = bardgroup;
                case 'logistic'
                    logmodel = logistic(yTrain, XTrain, 0.001);
                    model = logmodel;
                case 'bayeslog'
                    bayeslogmodel = bayeslog(yTrain, XTrain);
                    model = bayeslogmodel;
                case 'bardlog'
                    bardlogmodel = bardlog(yTrain, XTrain);
                    model = bardlogmodel;
                case 'bardloggroup'
                    bardloggroup = bardlog_grouped(yTrain, XTrain, NG);
                    model = bardloggroup;
                otherwise
                    disp('unknown method');
            end
        end

        for m = 1:numMethods
            method = methods{m};
            switch(method)
                case 'flda'
                    yPredict = XTest*fisher.b + fisher.b0;
                case 'swlda'
                    yPredict = XTest*swlda.b;
                case 'svm'
                    [predict_label,predict_accuracy,predict_decvalue] = svmpredict(yTest, XTest, svmmodel);
                    yPredict = svmmodel.Label(1)*predict_decvalue;
                    % yPredict = predict_decvalue;
                case 'ridge'
                    yPredict = XTest*ridge.b + ridge.b0;
                case 'lasso'
                    yPredict = XTest*lasso.b + lasso.b0;
                case 'lassogroup'
                    yPredict = XTest*lassogroup.b + lassogroup.b0;
                case 'bayes'
                    yPredict = XTest*bayes.b + bayes.b0;
                case 'bard'
                    yPredict = XTest*bard.b + bard.b0;
                case 'bardgroup'
                    yPredict = XTest*bardgroup.b + bardgroup.b0;
                case 'logistic'
                    Y_predict = X(:,feature_selected)*logmodel.b + logmodel.b0;
                case 'bayeslog'
                    Y_predict = X(:,feature_selected)*bayeslogmodel.b + bayeslogmodel.b0;
                case 'bardlog'
                    Y_predict = X(:,feature_selected)*bardlogmodel.b + bardlogmodel.b0;
                case 'bardloggroup'
                    Y_predict = X(:,feature_selected)*bardloggroup.b + bardloggroup.b0;
                otherwise
                    disp('unknown method');
            end
            accuracy(fold, p, m) = length(find(sign(yPredict) == sign(yTest)))/numTest;
        end
    end
end

%% visualize
figure;
plotstyle = {'b','k','r','b--','k--','r--'};
hold on; grid on;
for m = 1:numMethods
    plot(log10(params), squeeze(mean(accuracy(:,:,m), 1))*100,plotstyle{m},'LineWidth',2);
    axis([min(log10(params)) max(log10(params)) 0 100]);
    xlabel('log_{10} (\lambda)');
    ylabel('Accuracy (%)');
end
h = legend(methods{1},methods{2},methods{3});
set(h,'Location','SouthEast');