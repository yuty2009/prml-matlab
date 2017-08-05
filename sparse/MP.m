%% Matching Pursuit
% x: the signal that needs a sparse representation (N by 1)
% D: the dictionary (N by P), P is the number of basis
% k: k-sparse representation
function [varargout] = MP(x,D,k)

[N,P] = size(D);

rs = x;
w = zeros(P,1);
for i = 1:k
    coef = D'*rs;
    [maxv,maxi] = max(abs(coef));
    idx(i) = maxi;
    w(maxi) = D(:,maxi)'*rs;
    rs = x - D*w;
end
yf = D*w;
mse = rs.^2/N;

varargout{1} = w;
if nargout > 1, varargout{2} = yf; end
if nargout > 2, varargout{3} = mse; end