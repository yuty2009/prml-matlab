%% Least square regression with norm1 regularization
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% b: P by 1 regression coefficients
% b0: the intercept
%
% Lasso regularization can be re-formulated as
% g(x)  =  L(x)  +  \lambda \sqrt {x^T x + \epsilon}
% In such a case, the later item is differentiable and the optimzation
% problem can be solved by an iterated algorithm
function [varargout] = lassoreg(y,X,lambda)

if nargin < 3, lambda = 1e-4; end

PHI = cat(2, ones(size(X,1),1), X); % add a constant column to cope with bias
[N, P] = size(PHI);

% if (N == rank(PHI)) % underdetermined
%     f = lambda*ones(2*P,1);
%     Aeq = [PHI,-PHI];
%     beq = y;
%     lb = zeros(2*(P),1);
%     [w1,fval,exitflag] = linprog(f,[],[],Aeq,beq,lb,[]);
%     if (1)
%         w = w1(1:P) - w1(P+1:2*P);
%     end
% else % overdetermined
    cvx_quiet(false);
    cvx_begin
        variable w(P,1);
        minimize norm(y-PHI*w) + lambda*norm(w, 1);
    cvx_end
% end

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