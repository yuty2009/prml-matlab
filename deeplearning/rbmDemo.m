clc
clear

% datapath = 'f:\prmldata\mnist\';
datapath = '/Users/n0n/work/data/prmldata/mnist/';
imagefname = [datapath 'train-images-idx3-ubyte'];
labelfname = [datapath 'train-labels-idx1-ubyte'];

images = loadMNISTImages(imagefname);
X = images';
[N,P] = size(X);

opts.maxepochs = 1;
opts.stopeps = 1e-3;
opts.batchsize = 100;
S2 = 100;
rbm = rbminit(P,S2,'zero');
rbm.momentum = 0.5; % 0.9 after 5 epoches
rbm = rbmtrain(rbm,X,opts);

displayImageGrid(rbm.W,12); 