clc
clear

load faithfull.txt;

X = faithfull(:,2:end);
[idx,ctrs] = kmeans(X,2);

figure;

subplot(121);
scatter(faithfull(find(faithfull(:,1)==1),2), faithfull(find(faithfull(:,1)==1),3), 'r.');
hold on;
scatter(faithfull(find(faithfull(:,1)==2),2), faithfull(find(faithfull(:,1)==2),3), 'b.');

subplot(122);
scatter(X(idx == 1,1), X(idx == 1,2), 'r.');
hold on;
scatter(X(idx == 2,1), X(idx == 2,2), 'b.');
hold on;
scatter(ctrs(:,1), ctrs(:,2), 'k+');