%% Multi-class Fisher's Linear Discriminant Analysis
%  t: N by 1 class labels
%  X: N by P matrix, N observation of P dimensional feature vectors
function W = FLDAM(t,X,mode,lambda)

if nargin <= 2
    mode = '1vR';
    lambda = 1e-4;
elseif nargin <= 3
    lambda = 1e-4;
end

[N,P] = size(X);

labels = unique(t);
labels = sort(labels);
K = length(labels);

switch(mode)
    case '1vR' % one-versus-the-rest
        W = zeros(P+1,K);
        for i = 1:K
            idx1 = find(t==labels(i)); % one
            idx0 = setdiff(1:N,idx1); % the rest
            tt = t;
            tt(idx1) = 1;
            tt(idx0) = -1;
            [b,b0] = FLDA(tt,X,lambda);
            W(:,i) = [b0;b];
        end
    case '1v1' % one-versus-one
        W = zeros(P+1,K*(K-1)/2);
        index = 1;
        for i = 1:K-1
            idx1 = find(t==labels(i)); % one
            N1 = length(idx1);
            for j = i+1:K
                idx2 = find(t==labels(j)); % another one
                N2 = length(idx2);
                XX = X([idx1,idx2],:);
                tt = t([idx1,idx2]);
                tt(1:N1) = 1;
                tt(N1+(1:N2)) = -1;
                [b,b0] = FLDA(tt,XX,lambda);
                W(:,index) = [b0;b];
                index = index + 1;
            end
        end
    otherwise
        disp('unknown mode');
end

