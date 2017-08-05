function [fdr] = fisherratio(y, X)
% y: N by 1 labels
% X: N samples by M features

[N, M]=size(X);

c1 = max(y);
c2 = min(y);
idx1 = find(y == c1);
idx2 = find(y == c2);

fdr = zeros(M,1);

for i = 1:M
    mean1 = mean(X(idx1,i));
    mean2 = mean(X(idx2,i));
    var1 = var(X(idx1,i));
    var2 = var(X(idx2,i));
    fdr(i) = (mean1-mean2)^2/(var1+var2);
end