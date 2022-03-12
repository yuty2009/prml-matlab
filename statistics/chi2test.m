%% Hypotheis test for a one-of-k selection problem
% x: number of correct classified samples
% n: number of total collected samples
% k: number of classes
% alpha: significance level
function [h,p] = chi2test(x,n,k,alpha,tail)

if nargin < 4, alpha = 0.05; end
if nargin < 5, tail = 'both'; end

beta = n/k;
x2 = (x-beta)^2/beta + (beta-x)^2/((k-1)*beta);
sgn = sign(x-beta);
switch(tail)
    case 'both'
        p = 1 - chi2cdf(x2,1); % note that DoF is 1 here
        p = min(2*p,1);
    case 'left'
        p = 1 - chi2cdf(x2,1);
        if sgn > 0
            p = 1 - p;
        end
    case 'right'
        p = 1 - chi2cdf(x2,1); % note that DoF is 1 here
        if sgn < 0
            p = 1 - p;
        end
end

if nargout > 1, h = (p <= alpha); end
