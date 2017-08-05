clc
clear


N1 = 500;
N2 = 500;

r1 = sqrt(abs(randn(N1,1))); % Radius
a1 = 2*pi*rand(N1,1);  % Angle
X1 = [r1.*cos(a1), r1.*sin(a1)]; % Points
y1 = ones(N1,1);

r2 = sqrt(abs(3*randn(N2,1))+2); % Radius
a2 = 2*pi*rand(N2,1);      % Angle
X2 = [r2.*cos(a2), r2.*sin(a2)]; % points
y2 = -ones(N2,1);

t = [-5 5];


figure;
subplot(121);
hold on
plot(X1(:,1),X1(:,2),'r.','MarkerSize',10)
plot(X2(:,1),X2(:,2),'b.','MarkerSize',10)
axis([-5 5 -5 5]);

X = cat(1,X1,X2);
y = cat(1,y1,y2);

NC = 100;
abmodel = adaboost_train(y,X,@ftrain,@ftest,NC);

for i = 1:NC
    model1 = abmodel.models{i};
    w1 = model1.SVs' * model1.sv_coef; % for SVM
    b1 = -model1.rho;
 
    yt = (-w1(1)*t - b1)/w1(2);
    plot(t,yt);
    % pause;
end

[ypred] = adaboost_test(abmodel,X,@ftest);

index1 = find(ypred==1);
index2 = find(ypred==-1);

subplot(122);
hold on
plot(X(index1,1),X(index1,2),'r.','MarkerSize',10)
plot(X(index2,1),X(index2,2),'b.','MarkerSize',10)
axis([-5 5 -5 5]);
