%% FLDA backward model example
% Reference: On the interpretation of weight vectors of linear models in
% multivariate neuroimaging, Neuroimage, 2014

clc
clear

M = 2;
N1 = 100;
N2 = 100;
MU1 = [1.5;0];
MU2 = [-1.5;0];
SIGMA = [1.02,-0.30;-0.30,0.15];
X1 = mvnrnd(MU1,SIGMA,N1);
X2 = mvnrnd(MU2,SIGMA,N2);

X = cat(1, X1, X2);
y = [ones(N1,1); -1*ones(N2,1)];
N = length(y);

svmoption = '-s 0 -t 0 -c 1';
model0 = svmtrain(y,X,svmoption);
w0 = model0.SVs' * model0.sv_coef;
b0 = -model0.rho;

NP = 5000;
ws = zeros(NP,M);
for i = 1:NP
    perm = randperm(N);
    y1 = y(perm);
    model1 = svmtrain(y1,X,svmoption);
    w1 = model1.SVs' * model1.sv_coef;
    ws(i,:) = w1;
end

% test
t1 = (w0(1)-mean(ws(:,1)))/std(ws(:,1));
t2 = (w0(2)-mean(ws(:,2)))/std(ws(:,2));
[h1,p1] = permtest(w0(1),ws(:,1));
[h2,p2] = permtest(w0(2),ws(:,2));

t = min(X(:,1)):0.1:max(X(:,1));
v1 = (-w0(1)*t-b0)/w0(2);

figure;
hold on;
scatter(X1(:,1),X1(:,2),'bx');
scatter(X2(:,1),X2(:,2),'ro');
plot(t,v1,'g-');
axis equal;
grid on;
legend('c1','c2','SVM');
