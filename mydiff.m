%DIFF Difference and approximate derivative.
function Y = mydiff(X, dim)

dims = size(X);
N = length(dims);
if nargin < 2, dim = 1; end
if dim > N, error('dim too large'); end

U1 = diffmat1(dims(dim));

if (dim == 1)
    Y = U1*X;
else
    X1 = permute(X, [dim 2:(dim-1) 1 (dim+1):N]);
    Y1 = U1*X1;
    Y = permute(Y1, [dim 2:(dim-1) 1 (dim+1):N]);
end