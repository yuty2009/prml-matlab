clc
clear

load('pcadata.txt');

X = pcadata';
figure(1);
scatter(X(:,1), X(:,2));
title('Raw data');

[PC,S] = BPCA(X);
hold on
plot([0 PC(1,1)], [0 PC(2,1)]);
plot([0 PC(1,2)], [0 PC(2,2)]);

Xrot = X*PC;
figure(2);
scatter(Xrot(:,1), Xrot(:,2));
title('Xrot');

k = 1;
Xpc = X*PC(:,1:k);
Xpc(:,k+1:size(X,2)) = 0;
Xrecon = Xpc*PC';
figure(3);
scatter(Xrecon(:,1), Xrecon(:,2));
title('Xrecon');