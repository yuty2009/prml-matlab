%% crossvalidation demo for regression
clc
clear

%% generate train dataset
N = 256;
P = 1024; % feature dimension
NG = 32; % number of groups
PG = floor(P/NG); % number of feature per-group
groups = ceil([1:P]'/PG);
NSG = 10; % number of active groups
perm = randperm(NG);
actives = perm(1:NSG);
X = randn(N,P);
w0 = zeros(P,1);
for i = 1:NSG
    w0(find(groups==actives(i))) = ...
      randn(size(w0(find(groups==actives(i))))); % gaussian signal
%   w0(find(groups==actives(i))) = ...
%       ones(size(w0(find(groups==actives(i))))); % uniform signal
end
sigma = 0.2;
y = X*w0 + sigma*randn(N,1);
save('grouplasso_gaussian', 'X', 'y', 'NG', 'w0');

% load('grouplasso_uniform');

%% perform crossvalidation
numFolds = 10;
numTotal = length(y);
numTest = floor(numTotal/numFolds);
numTrain = numTotal - numTest;
indexTotal = randperm(numTotal); % [1:numTotal]; %

methods = {'ridge','lasso','lassogroup'};
% methods = {'ridge','lasso','lassogroup','bayes','bard','bardgroup'};
numMethods = length(methods);

params = logspace(-10,5,20);%1e-2;%
numParams = length(params);

disp([num2str(numFolds) ' fold cross-validation']);
rmse = zeros(numFolds, numParams, numMethods);
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
                    lassogroup = lassogroupreg(yTrain, XTrain, NG, lambda);
                    model = lassogroup;
                case 'bayes'
                    bayes = bayesreg(yTrain, XTrain);
                    model = bayes;
                case 'bard'
                    bard = bayesard(yTrain, XTrain);
                    model = bard;
                case 'bardgroup'
                    bardgroup = bardgroupreg(yTrain, XTrain, numChannels);
                    model = bardgroup;
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
                otherwise
                    disp('unknown method');
            end
            rmse(fold, p, m) = norm(yPredict-yTest);
        end
    end
end

%% visualize
figure;
plotstyle = {'b','k','r','b--','k--','r--'};
hold on; grid on;
for m = 1:numMethods
    plot(log10(params), squeeze(mean(rmse(:,:,m), 1)),plotstyle{m},'LineWidth',2);
    xlim([min(log10(params)) max(log10(params))]);
    xlabel('log_{10} (\lambda)');
    ylabel('rmse');
end
h = legend(methods{1},methods{2},methods{3});
set(h,'Location','SouthEast');
saveas(gcf, 'cvdata_crossvalidation_lambda.fig');