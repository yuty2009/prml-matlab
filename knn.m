%% K-nearest neighbor classifier
function y = knn(X,XTrain,yTrain,K)

[N1,P] = size(X);
[N2,P] = size(XTrain);
labels = unique(yTrain);
NC = length(labels);

y = zeros(N1,1);

for i = 1:N1
    x1 = X(i,:);
    D1 = zeros(N2,1);
    for j = 1:N2
        x2 = XTrain(j,:);
        D1(j) = norm(x1-x2);
    end
    
    [dummy,idx] = sort(D1,'ascend');
    idxK = idx(1:K);
    numK = zeros(NC,1);
    for j = 1:NC
        numK(j) = length(find(yTrain(idxK)==labels(j)));
    end
    [dummy,idxT] = max(numK);
    y(i) = labels(idxT);
end