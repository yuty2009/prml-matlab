clc
clear

N = 500;
K = 3;
MU = [3 2;0 -1;-3 -5];
SIGMA = cat(3,[1 0;0 4],[4 0;0 1],[1 0;0 4]);
Pz = ones(1,K)/K;

X = gmmrnd(N, MU, SIGMA, Pz);
[z,Pzgx] = gmmposterior(X, MU, SIGMA, Pz);

[MU1,SIGMA1,Pz1] = gmmfit(X,K);
[z1,Pzgx1] = gmmposterior(X, MU1, SIGMA1, Pz1);

index1 = find(z==1);
index2 = find(z==2);
index3 = find(z==3);

index11 = find(z1==1);
index12 = find(z1==2);
index13 = find(z1==3);

D = zeros(N,K);
D(:,1) = mahal(X,X(index1,:));
D(:,2) = mahal(X,X(index2,:));
D(:,3) = mahal(X,X(index3,:));

figure;

% raw data
subplot(221);
scatter(X(:,1),X(:,2),'o');
colorbar;
title('raw data');

% cluster
subplot(222);
hold on;
scatter(X(index1,1),X(index1,2),'ro');
scatter(X(index2,1),X(index2,2),'go');
scatter(X(index3,1),X(index3,2),'bo');
error_ellipse(squeeze(SIGMA(:,:,1)), mean(X(index1,:)));
error_ellipse(squeeze(SIGMA(:,:,2)), mean(X(index2,:)));
error_ellipse(squeeze(SIGMA(:,:,3)), mean(X(index3,:)));
colorbar;
title('cluster');
legend('c1','c2','c3','Location','NW')

% posterior
subplot(223);
scatter(X(:,1),X(:,2),100,Pzgx(:,2),'o');
colorbar;
title('posterior');

% Mahalanobis distance
% subplot(224);
% scatter(X(:,1),X(:,2),100,D(:,2),'o');
% colorbar;
% title('Mahalanobis distance');

% fit plot
subplot(224);
hold on;
scatter(X(index11,1),X(index11,2),'ro');
scatter(X(index12,1),X(index12,2),'go');
scatter(X(index13,1),X(index13,2),'bo');
error_ellipse(squeeze(SIGMA1(:,:,1)), mean(X(index11,:)));
error_ellipse(squeeze(SIGMA1(:,:,2)), mean(X(index12,:)));
error_ellipse(squeeze(SIGMA1(:,:,3)), mean(X(index13,:)));
colorbar;
title('fitting');
legend('c1','c2','c3','Location','NW')