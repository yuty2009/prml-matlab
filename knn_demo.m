clc
clear

load tulip;

[P,N] = size(X);

xmin1 = min(X(1,:));
xmin2 = min(X(2,:));
xmax1 = max(X(1,:));
xmax2 = max(X(2,:));
x1 = xmin1:0.5:xmax1;
x2 = xmin2:0.5:xmax2;
for i = 1:length(x1)
    for j = 1:length(x2)
        X1(:,(i-1)*length(x1)+j) = [x1(i); x2(j)];
    end
end

y1 = knn(X1',X',y,10);

idx11 = find(y==1);
idx12 = find(y==2);
idx21 = find(y1==1);
idx22 = find(y1==2);

figure;
hold on;
scatter(X(1,idx11),X(2,idx11),50,'ro');
scatter(X(1,idx12),X(2,idx12),50,'b+');
% scatter(X1(1,idx21),X1(2,idx21),2,'r.');
scatter(X1(1,idx22),X1(2,idx22),2,'b.');