%% Ridge regression
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% b: P by 1 regression coefficients
% b0: the intercept
function [varargout] = ridgereg(y, X, lambda)

if nargin < 3, lambda = 1e-4; end

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N, P] = size(PHI);

if (P > N)
    invC = woodburyinv(lambda*eye(P),PHI',PHI,eye(N));
else
    invC = (lambda*eye(P)+ PHI'*PHI)^(-1);
end
w = invC*PHI'*y;

b = w(2:end);
b0 = w(1);

if nargout == 1
    model.b = b;
    model.b0 = b0;
    varargout{1} = model;
elseif nargout == 2
    varargout{1} = b;
    varargout{2} = b0;
end