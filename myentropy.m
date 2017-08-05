%% Estimate entropy
% X is an N by M stocastic variable, M = {1,2}
% M = 1: univariate entropy
% M = 2: joint entropy of two variables
% When M > 2, it is the same as entropy.m for a 2-D image
function E = myentropy(varargin)

nbins = 100;
X = varargin{1};
[N,M] = size(X);
if (nargin) >= 2
    nbins = varargin{2};
end

X = double(X);
if (M==1)
    p = hist(X,nbins);
elseif (M==2)
    p = hist3(X,[nbins nbins]);
    p = p(:);
else
    X = X(:);
    p = hist(X,nbins);
end
% remove zero entries in p 
p(p==0) = [];
% normalize p so that sum(p) is one.
p = p ./ sum(p);
E = -sum(p.*log2(p));