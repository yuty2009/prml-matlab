%% Regularized Linear Discriminant Analysis
% y: N by 1 labels 
% X: N by P matrix, N observation of P dimensional feature vectors 
% Reference: B. Blankertz, S. Lemm, M. Treder, S. Haufe, and K.R. Muller,
% Single-trial analysis and classification of ERP components--a tutorial,
% Neuroimage, vol. 56, (no. 2), pp. 814-25, 2011-05-15 2011.
function [varargout] = RLDA(t,X)

[N,P] = size(X);

index1 = find(t==1);
index2 = find(t==-1);
N1 = length(index1);
N2 = length(index2);

X1 = X(index1,:);
X2 = X(index2,:);
mX = mean(X,1);
mu1 = mean(X1,1);
mu2 = mean(X2,1);

Sw = cov(X);

v = trace(Sw)/P;
z = zeros(N,P,P);
for i = 1:N
    z(i,:,:) = (X(i,:)-mX)'*(X(i,:)-mX);
end
varZ = var(z,0,1);
diagS = diag(Sw);
squareS = Sw.^2;
denomi = sum(squareS(:))-sum(diag(squareS))+sum((diagS-v).^2);
gamma = (N/(N-1)^2)*sum(varZ(:))/denomi;

b = inv((1-gamma)*Sw+gamma*v*eye(size(Sw)))*(mu1-mu2)';
b0 = -(mu1+mu2)*b/2;

if nargout == 1
    model.b = b;
    model.b0 = b0;
    varargout{1} = model;
elseif nargout == 2
    varargout{1} = b;
    varargout{2} = b0;
elseif nargout == 3
    varargout{1} = b;
    varargout{2} = b0;
    
    pm1 = mu1*b;
	pm2 = mu2*b;
    fdr = (pm1-pm2)^2/(b'*Sw*b);
    varargout{3} = fdr;
end