clc
clear

datapath = 'e:\prmldata\mnist\';
% datapath = '/Users/n0n/work/data/prmldata/mnist/';
images = loadMNISTImages([datapath 'train-images-idx3-ubyte']);
labels = loadMNISTLabels([datapath 'train-labels-idx1-ubyte']);
labels(labels==0) = 10; % Remap 0 to 10

X = images';
t = labels;

lambda = 1e-4;
W = softmax1(t,X,lambda);
% svmoption = ['-s 0 -t 0 -c 1 -g 0.001'];
% model = svmtrain(t,X,svmoption);
% W = FLDAM(t,X);

images = loadMNISTImages([datapath 't10k-images-idx3-ubyte']);
labels = loadMNISTLabels([datapath 't10k-labels-idx1-ubyte']);
labels(labels==0) = 10; % Remap 0 to 10

X = images';
y = softmaxpredict(X,W);
% y0 = zeros(size(X,1),1);
% [y,dummy1,dummy2] = svmpredict(y0,X,model);
% y = FLDAMpredict(X,W);

acc = mean(labels(:) == y(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);
