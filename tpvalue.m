%% Calculate the hypothesis p-value
% y: N by 1 labels
% X: N samples by M features
function [pvalue] = tpvalue(y, X)

[N, M]=size(X);

c1 = max(y);
c2 = min(y);
idx1 = find(y == c1);
idx2 = find(y == c2);

pvalue = zeros(M,1);

for i = 1:M
    x1 = X(idx1,i);
    x2 = X(idx2,i);
    [h,p,ci,stats] = ttest2(x1,x2,0.05,'both');
    pvalue(i) = p;
end