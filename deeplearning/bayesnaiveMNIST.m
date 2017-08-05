clc
clear

datapath = 'e:\prmldata\mnist\';
% datapath = '/Users/n0n/work/data/prmldata/mnist/';
images = loadMNISTImages([datapath 'train-images-idx3-ubyte']);
labels = loadMNISTLabels([datapath 'train-labels-idx1-ubyte']);
labels(labels==0) = 10; % Remap 0 to 10

inputtype = 'binary';

X = images';
if strcmpi(inputtype,'binary')
    m = mean(X(:));
    X = double(X>m); % binarize the data
end
t = labels;

% Training
nbmodel = bayesnaive(t,X,inputtype);

% Generate samples from posterior predictive distribution
Nsim = 10;
Npixels = size(X,2);
Xsamp = [];
for i = 1:10 % 10 classes
    switch(inputtype)
        case 'binary'
            px = nbmodel.theta(i,:);
            Xtemp = rand(Nsim,Npixels) < repmat(px,Nsim,1);
        case 'gauss'
            Xtemp = mvnrnd(nbmodel.mu(i,:),nbmodel.sigma(i,:),Nsim);
    end
    Xsamp = cat(1,Xsamp,Xtemp);
end
displayImageGrid(Xsamp',12);

% Predicting
images = loadMNISTImages([datapath 't10k-images-idx3-ubyte']);
labels = loadMNISTLabels([datapath 't10k-labels-idx1-ubyte']);
labels(labels==0) = 10; % Remap 0 to 10

X = images';
if strcmpi(inputtype,'binary')
    m = mean(X(:));
    X = double(X>m); % binarize the data
end
y = bayesnaivepredict(X,nbmodel);

acc = mean(labels(:) == y(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);
