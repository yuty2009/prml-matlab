%% Vector Quantization using LBG's algorithm
%% reference: http://www.data-compression.com/vq.shtml
% input params
% X: input matrix, N by M, sample by feature
% K: desired number of codevectors
% epsilon: distortion threshold
% output params
% C: codebook
% Q: N by 1, code of the samples
function [C,Q] = LBGVQ(X,K,epsilon)

[N,P] = size(X);

C = zeros(K,P);
Q = zeros(N,1);

k = 1;
C(k,:) = mean(X);
rmse = (1/(N*P))*sum(sum((X-ones(N,1)*C(k,:)).^2));


while k < K
    
    for i = 1:k
        C(i,:) = (1+epsilon)*C(i,:);
        C(k+i,:) = (1-epsilon)*C(i,:);
    end
    k = 2*k;
    
    disp(['No. of codevectors is ' num2str(k)]);
    
    drmse = Inf;
    while drmse > epsilon
        rmseold = rmse;
        
        D = zeros(N,k);
        for i = 1:N
            for j = 1:k
                D(i,j) = sum((X(i,:) - C(j,:)).^2);
            end
            [dummy Q(i)] = min(D(i,:));
        end

        rmse = 0;
        for i = 1:k
            iC = find(Q==i);
            C(i,:) = mean(X(iC,:));
            rmse = rmse + sum(sum((X(iC,:)-ones(length(iC),1)*C(i,:)).^2));
        end
        rmse = rmse/(N*P);
        
        drmse = (rmseold-rmse)/rmseold;
    end
    
end
