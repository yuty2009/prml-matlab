%% Gaussian Discrimant Analysis
% Page 198-203 of PRML
function [phi,mu1,mu2,Sigma] = GDA(y,X)

N = length(y);

c1 = max(y);
c2 = min(y);
idx1 = find(y==c1);
idx2 = find(y==c2);
N1 = length(idx1);
N2 = length(idx2);

% maximum likelihood estimation
phi = N1/(N1+N2);
mu1 = mean(X(idx1,:));
mu2 = mean(X(idx2,:));
cov1 = cov(X(idx1,:));
cov2 = cov(X(idx2,:));
Sigma = (N1/(N1+N2))*cov1 + (N2/(N1+N2))*cov2;