clc
clear

X = gencircledata([1;1],5,250,1)';

opts.ktype = 'rbf';
opts.args = [4,0];
model = kPCA(X,opts);

XR = kPCAproj(X,2,model);

figure;
hold on;
scatter(X(:,1),X(:,2),'bx');
scatter(XR(:,1),XR(:,2),'ro');
legend('raw','recon');
