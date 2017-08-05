%% Fisher's Linear Discriminant Analysis
% y: N by 1 labels
% X: N by P matrix, N observation of P dimensional feature vectors
function [varargout] = FLDA(y,X,lambda)

if nargin <= 2
    lambda = 1e-4;
end

[N,P] = size(X);

index1 = find(y==1);
index2 = find(y==-1);
N1 = length(index1);
N2 = length(index2);

X1 = X(index1,:);
X2 = X(index2,:);

mu1 = mean(X1,1);
mu2 = mean(X2,1);

% % subtract off the mean for each dimension
% mX1matrix = repmat(mu1, N1, 1); 
% mX2matrix = repmat(mu2, N2, 1); 
% 
% % Sw will be singular due to:
% % 1, the number of samples is less than that of variables/features
% % 2, rank(X-mean(X)) = rank(X)-1
% Sw = (X1-mX1matrix)'*(X1-mX1matrix)/N1 ...
%     + (X2-mX2matrix)'*(X2-mX2matrix)/N2;

Sw = cov(X);

b = inv(Sw+lambda*eye(size(Sw)))*(mu1-mu2)';
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
    
    pm1 = mX1*b;
	pm2 = mX2*b;
    fdr = (pm1-pm2)^2/(b'*Sw*b);
    varargout{3} = fdr;
end