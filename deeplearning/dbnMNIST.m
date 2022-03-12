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
SN = [P,S1,S1,10];

dbn1 = dbninit(2,SN);
dbn1.rbm{1}.lrate = 0.1;
dbn1.rbm{1}.momentum = 0.5;
dbn1.rbm{2}.lrate = 0.1;
dbn1.rbm{2}.momentum = 0.5;
dbn1.oTF = 'softmax';
opts.lambda = 1e-4;
opts.verbose = 'on';
opts.maxepochs = 10;
% opts.optmethod = 'lbfgs';
opts.method = 'minibatch';
opts.batchsize = 50;
opts.lrate = 1;
dbn1 = dbntrain(dbn1,y,X,opts);
displayImageGrid(dbn1.rbm{1}.W,12); 

% fine tune
disp('Fine tuning');
dbn2 = mlptrain(dbn1,y,X,opts);
save('saves/dbn2', 'dbn2');

% test
testData = loadMNISTImages([datapath 't10k-images-idx3-ubyte']);
testLabels = loadMNISTLabels([datapath 't10k-labels-idx1-ubyte']);
testLabels(testLabels == 0) = 10; % Remap 0 to 10

pred = mlppredict(dbn1,testData');

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

pred = mlppredict(dbn2,testData');

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
