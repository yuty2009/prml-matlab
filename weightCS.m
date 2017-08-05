%% Channel selection via ranking of classifier weights
%     y: N by 1 class labels
%     X: N by P feature matrix
%     NC: number of available channels
%     NS: number of channels to be selected
%     CS: indices of selected channels
%     FS: indices of selected features
function [CS,FS] = weightCS(w,NC,NS)

P = length(w);
P0 = P/NC;
IM = reshape(1:P,P0,NC); % indices in matrix form

wAvg = mean(reshape(abs(w),P0,NC));
[dummy,index] = sort(wAvg,'descend');

CS = index(1:NS);
FS = IM(:,CS);
FS = FS(:);