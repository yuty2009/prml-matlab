clc
clear

datapath = 'f:\prmldata\mnist\';
% datapath = '/Users/n0n/work/data/prmldata/mnist/';
mnistData   = loadMNISTImages([datapath 'train-images-idx3-ubyte']);
mnistLabels = loadMNISTLabels([datapath 'train-labels-idx1-ubyte']);

% Simulate a Labeled and Unlabeled set
labeledSet   = find(mnistLabels >= 0 & mnistLabels <= 4);
unlabeledSet = find(mnistLabels >= 5);

numTrain = round(numel(labeledSet)/2);
trainSet = labeledSet(1:numTrain);
testSet  = labeledSet(numTrain+1:end);

unlabeledData = mnistData(:, unlabeledSet);

trainData   = mnistData(:, trainSet);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5

testData   = mnistData(:, testSet);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5

% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n', size(testData, 2));

opts.lambda = 3e-3;
opts.beta = 3;
opts.sparsity = 0.1;
opts.inittype = 'uniform';
opts.display = 'on';
opts.maxit = 400;
opts.optmethod = 'lbfgs';
ae = mlpinit(size(trainData,1),[196,size(trainData,1)],{'sigmoid','sigmoid'},'uniform');
ae = aetrain(ae,unlabeledData',opts);

displayImageGrid(ae.W{1},12);

trainFeatures = aeencode(ae,trainData');
testFeatures = aeencode(ae,testData');

lambda = 1e-4;
W = softmax(trainLabels',trainFeatures,lambda);

[pred] = softmax_predict(testFeatures,W);

fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));
