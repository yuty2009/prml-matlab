%% Compute R square
% y: N by 1 labels
% X: N samples by [P1,P2,...,PN] features
function [rr] = rsquare(y, X)

dims = size(X);

NF = prod(dims(2:end));
rr = zeros(NF,1);
X1 = X(:,:);
for i = 1:NF
    rr(i) = corr(y, X1(:,i))^2;
end
if length(dims)>2, rr = reshape(rr,dims(2:end)); end