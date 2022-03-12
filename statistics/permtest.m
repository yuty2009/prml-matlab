%% permutation test
% x,y: two vectors, with equal or unequal length, to be tested
%      or if x is a scalar, it tests whether the mean of y equal to x
% alpha: default 0.05
% tail: 'left', 'right', 'both', default 'both'
%       'both' ¡ª Means are not equal (two-tailed test). This is the default, when tail is unspecified.
%       'right' ¡ª Mean of x is greater than mean of y (right-tail test)
%       'left' ¡ª Mean of x is less than mean of y (left-tail test)
% NP: number of permutation resampling
function [h,p] = permtest(x,y,alpha,tail,NP)

if nargin <= 2, alpha = 0.05; end
if nargin <= 3, tail = 'both'; end
if nargin <= 4, NP = 5000; end

N1 = length(x);
N2 = length(y);

if N1 ~= 1
    t = mean(x) - mean(y);
    xy = zeros(N1+N2,1);
    xy(1:N1) = x;
    xy(N1+1:N1+N2) = y;

    t1 = zeros(NP,1);
    for i = 1:NP
        perm = randperm(N1+N2);
        x1 = xy(perm(1:N1));
        y1 = xy(perm(N1+1:N1+N2));
        t1(i) = mean(x1) - mean(y1);
    end
else
    t = x;
    t1 = y;
    NP = N2;
end

t2 = sort(t1,'ascend');

p1 = length(find(t2 <= t))/NP;
p2 = length(find(t2 >= t))/NP;
p = min(p1,p2);

switch(tail)
    case 'left'
        if t > 0
            p = 1 - p;
        end
    case 'right'
        if t < 0
            p = 1 - p;
        end
    case 'both'
        p = min(2*p,1);
end

if nargout > 1, h = (p <= alpha); end
