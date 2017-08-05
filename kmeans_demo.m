clc
clear

mu1 = [2 6];
mu2 = [6 2];
Sigma = [1 1.5; 1.5 3];

N1 = 50;
N2 = 50;
X1 = mvnrnd(mu1,Sigma,N1);
X2 = mvnrnd(mu2,Sigma,N2);

K = 2;
N = N1+N2;
X = cat(1,X1,X2);

%% [idx,C] = kmeans(X,2);
IDX = zeros(N,1);
C = X(1:K,:);

rmse = Inf;
drmse = Inf;
epsilon = 10^(-3);

i = 1;
while drmse > epsilon
    rmseold = rmse;
    
    rmse = 0;
    for n = 1:N
        for k = 1:K
            D(n,k) = norm(X(n,:)-C(k,:));
        end
        [dummy,IDX(n)] = min(D(n,:));    
        rmse = rmse + D(n,IDX(n));
    end
    
    % visualization
    idx1 = find(IDX==1);
    idx2 = find(IDX==2);

    hold off;
    scatter(X(idx1,1),X(idx1,2),'r*');
    hold on;
    scatter(X(idx2,1),X(idx2,2),'bo');
    scatter(C(1,1),C(1,2),200,'k+');
    scatter(C(2,1),C(2,2),200,'k+');
    
    % update
    for k = 1:K
        idxk = find(IDX==k);
        C(k,:) = mean(X(idxk,:));
    end

    drmse = abs(rmseold-rmse);
    disp(['Iteration ' num2str(i) ': rmse = ' num2str(rmse)]);
    i = i + 1;
    
    pause(2);
end


