clc
clear

NTR1 = 50; % train set size class 1
NTR2 = 50; % train set size class 2
NTE1 = 500; % test set size class 1
NTE2 = 500; % test set size class 2

NR = 30; % repetition
ND = 40; % max dimention

mu0 = [sqrt(0.5),sqrt(0.5)];

erate1 = zeros(NR,ND-1);
erate2 = zeros(NR,ND-1);
for i = 1:NR
    for j = 2:ND
        mu1 = [mu0, zeros(1,j-2)];
        mu2 = -mu1;
        
        XTR1 = mvnrnd(mu1,eye(j),NTR1);
        XTR2 = mvnrnd(mu2,eye(j),NTR2);
        XTR = cat(1,XTR1,XTR2);
        yTR = [ones(NTR1,1);-ones(NTR2,1)];
        model1 = bayeslog(yTR,XTR);
        model2 = blassoprobit(yTR,XTR);
        
        XTE1 = mvnrnd(mu1,eye(j),NTE1);
        XTE2 = mvnrnd(mu2,eye(j),NTE2);
        XTE = cat(1,XTE1,XTE2);
        yTE = [ones(NTE1,1);-ones(NTE2,1)];
        yP1 = sign(XTE*model1.b+model1.b0);
        yP2 = sign(XTE*model2.b+model2.b0);
        erate1(i,j-1) = length(find(yP1~=yTE))/(NTE1+NTE2);
        erate2(i,j-1) = length(find(yP2~=yTE))/(NTE1+NTE2);
    end
end

figure;
hold on;
plot(2:ND,mean(erate1),'b--');
plot(2:ND,mean(erate2),'r-');
plot([2,ND],[0.1587,0.1587],'k-.');
legend('non-sparse','sparse','optimal');
