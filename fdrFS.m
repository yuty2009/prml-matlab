%% Feature selection via ranking of Fisher ratios
%     y: N by 1 class labels
%     X: N by P feature matrix
%     NS: number of features to be selected
%     FS: indices of selected features
function FS = fdrFS(y,X,NS)

fdr = fisherratio(y,X);
[dummy,index] = sort(fdr,'descend');
FS = index(1:NS);