%% Least square regression with norm1 regularization
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% b: P by 1 regression coefficients
% b0: the intercept
% In this version, we use l1_ls function provided by Kwangmoo
% Koh, Seung-Jean Kim, and Stephen Boyd
% (http://www.stanford.edu/~boyd/l1_ls/).
% Faster implementations of lasso include Fix-point algorithm 
% (http://www.caam.rice.edu/~optimization/L1/fpc/) and SLEPl1 algorithm
% (http://www.public.asu.edu/~jye02/Software/slepl1/).
function [varargout] = lassoreg1(y,X,lambda)

if nargin < 3, lambda = 1e-4; end

PHI = cat(2, ones(size(X,1),1), X); % add a constant column to cope with bias
[N, P] = size(PHI);

[w,status,history] = l1_ls(PHI, y, lambda);

b = w(2:P);
b0 = w(1);

if nargout == 1
    model.b = b;
    model.b0 = b0;
    varargout{1} = model;
elseif nargout == 2
    varargout{1} = b;
    varargout{2} = b0;
end
