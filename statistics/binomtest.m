%% Hypotheis test for a one-of-k selection problem
% x: number of correct classified samples
% n: number of total collected samples
% k: number of classes
% alpha: significance level
function [h,p] = binomtest(x,n,k,alpha,tail)

if nargin < 4, alpha = 0.05; end
if nargin < 5, tail = 'both'; end

beta = 1/k;
z = (x/n-beta)/sqrt(beta*(1-beta)/(n+2.5));
switch(tail)
    case 'both'
        p = 1 - normcdf(abs(z),0,1);
        p = min(2*p,1);
    case 'left'
        p = normcdf(z,0,1);
        if z > 0
            p = 1 - p;
        end
    case 'right'
        p = 1 - normcdf(z,0,1);
        if z < 0
            p = 1 - p;
        end
end

if nargout > 1, h = (p <= alpha); end
