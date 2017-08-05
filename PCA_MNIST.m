clc
clear


datapath = 'e:\prmldata\mnist\';
trainData = loadMNISTImages([datapath 'train-images-idx3-ubyte']);
trainLabels = loadMNISTLabels([datapath 'train-labels-idx1-ubyte']);
[P,N] = size(trainData);
imagesize = sqrt(P);

PC = PCA(trainData(:,1:1000)');

Ms = [1 10 50 250];
K = length(Ms);
X1 = trainData(:,1);
Xs = zeros(P, K);
for i = 1:K
    X2 = X1'*PC(:,1:Ms(i));
    Xs(:,i) = PC(:,1:Ms(i))*X2';
end

figure;
subplot(1,K+1,1);
title('raw');
im = reshape(X1, imagesize, imagesize);
imshow(im);
for i = 1:K
    subplot(1,K+1,i+1);
    title(['m=' num2str(Ms(i))]);
    im = reshape(Xs(:,i), imagesize, imagesize);
    imshow(im);
end
