%% Least square regression with grouped lasso regularization
% The optimization objective is as following:
% w = {arg min}_w (||Xw-y||_2^2 + \lambda \sum_g^G ||w_{I_g}||_2)
% 
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% group: No. of groups or a group id vector
%        e.g. group = [1 1 1 2 2 2 3 3 3 4 4 4]' indicates that there are
%        4 group with 3 members in each
% b: P by 1 regression coefficients
% b0: the intercept
function [varargout] = glassoreg(y,X,group,lambda)

if nargin < 4, lambda = 1e-4; end

PHI = cat(2, ones(size(X,1),1), X); % add a constant column to cope with bias
[N, P] = size(PHI);

if (numel(group) == 1) group = ceil([1:size(X,2)]/group); end
group = [0;group(:)]; % account for bias
groupid = unique(group);
NG1 = length(groupid);
    
cvx_quiet(false);
cvx_begin
    variable w(P,1);
    expression funcstr;
    funcstr = ['(norm(w(find(group == groupid(' num2str(1) '))))'];
    for i = 2:NG1
        funcstr = [funcstr ' + norm(w(find(group == groupid(' num2str(i) '))))'];
    end
    funcstr = [funcstr ')'];
    minimize norm(y-PHI*w) + lambda*eval(funcstr);
cvx_end

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
