%% Feature selection via ranking of classifier weights
%     y: N by 1 class labels
%     X: N by P feature matrix
%     NF: number of available features
%     NS: number of features to be selected
%     FS: indices of selected features
function FS = weightFS(w,NS)

[dummy,index] = sort(w,'descend');
FS = index(1:NS);