clc
clear

datapath = 'e:\prmldata\mnist\';
% datapath = '/Users/n0n/work/data/prmldata/mnist/';
trainData   = loadMNISTImages([datapath 'train-images-idx3-ubyte']);
trainLabels = loadMNISTLabels([datapath 'train-labels-idx1-ubyte']);
trainLabels(trainLabels == 0) = 10; % Remap 0 to 10

y = trainLabels;
X = trainData';
[N,P] = size(X);
S1 = 196;
S2 = 196;
SN = [P,S1,S2,10];

sae1 = saeinit(2,SN);
sae1.ae{1}.beta = 3;
sae1.ae{1}.sparsity = 0.1;
sae1.ae{1}.lambda = 3e-3;
sae1.ae{1}.corruption = 0.5;
sae1.ae{2}.beta = 3;
sae1.ae{2}.sparsity = 0.1;
sae1.ae{2}.lambda = 3e-3;
sae1.ae{2}.corruption = 0.5;
sae1.oTF = 'softmax';
sae1.corruption = 0;
opts.lambda = 1e-4;
opts.verbose = 'on';
opts.maxepochs = 10;
% opts.optmethod = 'lbfgs';
opts.method = 'minibatch';
opts.batchsize = 50;
opts.lrate = 1;
sae1 = saetrain(sae1,y,X,opts);
displayImageGrid(sae1.ae{1}.W{1},12); 

% fine tune
disp('Fine tuning');
opts.lambda = [0,0,1e-4];
sae2 = mlptrain(sae1,y,X,opts);
save('tmp/sae2', 'sae2');

% test
testData = loadMNISTImages([datapath 't10k-images-idx3-ubyte']);
testLabels = loadMNISTLabels([datapath 't10k-labels-idx1-ubyte']);
testLabels(testLabels == 0) = 10; % Remap 0 to 10

pred = mlppredict(sae1,testData');

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

pred = mlppredict(sae2,testData');

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
