%% Multi-class Fisher's Linear Discriminant Component Analysis
%  Supervised dimension reduction
%  t: N by 1 class labels
%  X: N by P matrix, N observation of P dimensional feature vectors
function [W] = FLDCA(t,X,lambda)

if nargin <= 2
    lambda = 1e-4;
end

PHI = cat(2, ones(size(X,1),1),X); % add a constant column to cope with bias
[N,P] = size(PHI);

labels = unique(t);
K = length(labels);

mX = mean(PHI,1);

Sb = zeros(P);
Sw = zeros(P);
for k = 1:K
    idx = find(t==labels(k));
    Nk = length(idx);
    Xk = PHI(idx,:);
    mXk = mean(Xk,1);
    Sb = Sb + Nk*(mXk-mX)'*(mXk-mX);
    mXkmatrix = repmat(mXk,Nk,1); 
    Sw = Sw + (Xk-mXkmatrix)'*(Xk-mXkmatrix);
end

[V,D] = eig((Sw+lambda*eye(P))^(-1)*Sb);
[diagD,idx] = sort(diag(D),'descend');
idxW = find(diagD>0);
W = V(:,idx(idxW));
