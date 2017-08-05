%% Orthogonal Matching Pursuit
% x: the signal that needs a sparse representation (N by 1)
% D: the dictionary (N by P), P is the number of basis
% k: k-sparse representation
function [varargout] = OMP(x,D,k)

[N,P] = size(D);

rs = x;
w = zeros(P,1);
for i = 1:k
    coef = D'*rs;
    [maxv,maxi] = max(abs(coef));
    idx(i) = maxi;
    D1 = D(:,idx);
    w1 = (D1'*D1)^(-1)*D1'*x;
    rs = x - D1*w1;
end
w(idx) = w1;
yf = D1*w1;
mse = rs.^2/N;

varargout{1} = w;
if nargout > 1, varargout{2} = yf; end
if nargout > 2, varargout{3} = mse; end