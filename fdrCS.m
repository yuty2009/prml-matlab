%% Channel selection via ranking of Fisher ratios
%     y: N by 1 class labels
%     X: N by P feature matrix
%     NC: number of available channels
%     NS: number of channels to be selected
%     CS: indices of selected channels
%     FS: indices of selected features
function [CS,FS] = fdrCS(y,X,NC,NS)

[N,P] = size(X);
P0 = P/NC;
IM = reshape(1:P,P0,NC); % indices in matrix form

fdr = fisherratio(y,X);
fdrAvg = mean(reshape(fdr,P0,NC));
[dummy,index] = sort(fdrAvg,'descend');

CS = index(1:NS);
FS = IM(:,CS);
FS = FS(:);