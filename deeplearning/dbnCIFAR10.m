clc
clear

datapath = 'F:\prmldata\cifar-10\';
% datapath = '/Users/n0n/work/data/prmldata/cifar-10/';
trainData   = [];
trainLabels = [];
for i = 1:5
    load([datapath 'data_batch_' num2str(i)]);
    trainData = cat(1,trainData,data);
    trainLabels = cat(1,trainLabels,labels);
end
trainLabels(trainLabels == 0) = 10; % Remap 0 to 10

y = double(trainLabels);
X = double(trainData)/255;
displayColorImageGrid(X(1:100,:)');
[N,P] = size(X);

mX = mean(X);
epsilon = 0.1;	       % epsilon for ZCA whitening
[PC,S] = PCA(X);
ZCW = PC*diag(1./sqrt(diag(S)+epsilon))*PC';
X = X*ZCW;
displayColorImageGrid(X(1:100,:)');

S1 = 256;
S2 = 100;
SN = [P,S1,S2,10];
dbn1 = dbninit(2,SN);
dbn1.rbm{1}.lrate = 0.1;
dbn1.rbm{1}.momentum = 0.5;
dbn1.rbm{2}.lrate = 0.1;
dbn1.rbm{2}.momentum = 0.5;
dbn1.oTF = 'softmax';
opts.maxepochs = 10;
opts.stopeps = 1e-3;
opts.batchsize = 100;
opts.lambda = 1e-4;
opts.maxit = 200; % 
opts.verbose = 'on';
opts.optmethod = 'lbfgs';
dbn1 = dbntrain(dbn1,y,X,opts);
displayColorImageGrid(dbn1.rbm{1}.W);

% fine tune
disp('Fine tuning');
dbn2 = mlptrain(dbn1,y,X,opts);
save('saves/dbn2', 'dbn2');

% test
load([datapath 'test_batch']);
testData = data;
testLabels = labels;
testLabels(testLabels == 0) = 10; % Remap 0 to 10

yT = double(testLabels);
XT = double(testData)/255;
XT = XT*ZCW;

pred = mlppredict(dbn1,XT);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

pred = mlppredict(dbn2,XT);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
