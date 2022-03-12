function [varargout] = elasticnet(y, X, lambda, alpha)

PHI = cat(2, ones(size(X,1),1), X); % add a constant column to cope with bias
[N, P] = size(PHI);

% if (N == rank(PHI)) % underdetermined
%     H = lambda*eye(2*P);
%     f = alpha*ones(2*P,1);
%     Aeq = [PHI,-PHI];
%     beq = y;
%     lb = zeros(2*(P),1);
%     [w1,fval,exitflag] = quadprog(H,f,[],[],Aeq,beq,lb,[]);
%     if (1)
%         w = w1(1:P) - w1(P+1:2*P);
%     end
% else % overdetermined
    cvx_quiet(false);
    cvx_begin
        variable w(P,1);
        minimize norm(y-PHI*w) + lambda*norm(w,2) + alpha*norm(w,1);
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