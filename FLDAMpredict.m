%% Multi-class Fisher's Linear Discriminant Analysis
%  t: N by 1 class labels
%  X: N by P matrix, N observation of P dimensional feature vectors
function y = FLDAMpredict(X,W,mode)

if nargin <= 2
    mode = '1vR';
end

switch(mode)
    case '1vR' % one-versus-the-rest
        y = softmaxpredict(X,W);
    case '1v1' % one-versus-one
end