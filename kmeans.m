%% K-means clustering
% X: N by P data matrix, N samples with P dimention
% K: number of clusters
% IDX: N by 1 cluster labels
% C: K by P centroid locations
function [IDX,C] = kmeans(X,K,opts)

if nargin < 3
    opts.verbose = 0;
end

[N,P] = size(X);

IDX = zeros(N,1);
C = X(1:K,:);

rmse = Inf;
drmse = Inf;
epsilon = 10^(-6);

i = 1;
while drmse > epsilon
    rmseold = rmse;
    
    % update IDX
    rmse = 0;
    for n = 1:N
        for k = 1:K
            D(n,k) = norm(X(n,:)-C(k,:));
        end
        [dummy,IDX(n)] = min(D(n,:));    
        rmse = rmse + D(n,IDX(n));
    end
    
    % update C
    for k = 1:K
        idxk = find(IDX==k);
        C(k,:) = mean(X(idxk,:));
    end
    
    drmse = abs(rmseold-rmse);
    
    if isfield(opts,'verbose') && opts.verbose == 1
        disp(['Iteration ' num2str(i) ': rmse = ' num2str(rmse)]);
    end
    
    i = i + 1;
end