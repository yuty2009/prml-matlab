%% Principle Component Analysis
% X: N observation by P variables
% dim: number of principal components which would be reserved
function varargout = PCA(X)

[N,P] = size(X);

% preprocessing
% subtract off the mean for each variable
mX = mean(X);
X0 = X - repmat(mX,N,1);

if (N >= P)
    %% it may cause out of memory by calculation of covariance
    % calculate the covariance matrix
    covX = (X0'*X0)/N;
    [PC,S,V] = svd(covX);
else
    [dummy,D,PC] = svd(X0);
    D2 = D.^2;
    M = min([N,P]);
    S = zeros(P);
    S(1:M,1:M) = D2(1:M,1:M);
end

varargout{1} = PC;

if (nargout >= 2)
    varargout{2} = S;
end
