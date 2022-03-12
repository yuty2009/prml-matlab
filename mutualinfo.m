%% Estimate (normalized) mutual information
% y is N by 1 continous stocastic variables
% X is N by P with each column is a continous stocastic variables
% H(x|y) = H(x,y) - H(y)
% I(x,y) = H(x) - H(x|y) = H(y) - H(y|x) = H(x) + H(y) - H(x,y)
function [varargout] = mutualinfo(varargin)

nbins = 100;
y = varargin{1};
X = varargin{2};
[N,M] = size(X);
if (nargin) >= 3
    nbins = varargin{3};
end

for i = 1:M
    x = X(:,i);
    xy = [x,y];
    hx = myentropy(x,nbins);
    hy = myentropy(y,nbins);
    hxy = myentropy(xy,nbins);
    mi(i) = hx + hy - hxy;
    nmi(i) = 2*mi(i)/(hx+hy);
end

varargout{1} = mi;
if (nargout > 1)
    varargout{2} = nmi;
end