%% Variational logistic regression for binary classification 
% (Page 498-505 of PRML)
% Soluted through variational approximation & EM.
% 
% X: N by P design matrix with N samples of M features
% t: N by 1 target values 
% w: P+1 by 1 weight vector
function [varargout] = bvarlog(t,X)

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N, P] = size(PHI);
t(find(t==-1)) = 0; % the class label should be [1 0]

% initialization
w = ones(P,1); % rough initialization
an = 1e-9;
bn = 1e-9;
xi = ones(N,1);

% stop conditions
d_w = Inf;
maxit = 500;
stopeps = 1e-6;

i = 1;
while (d_w > stopeps)  && (i < maxit)
    wold = w;
    
    % E step
    E_alpha = an/bn;
    lambda = (1/(2*xi)).*(1/(1+exp(-xi))-1/2);
    sigma = (E_alpha*eye(P) + 2*PHI'*diag(lambda)*PHI)^(-1);
    w = sigma*PHI'*(t-1/2);
    % M step
    an = an + P/2;
    bn = bn + (1/2)*(trace(sigma+w*w')); 
    xi = diag(PHI*(sigma+w*w')*PHI');
    
    rmse = sum((t-PHI*w).^2);
    evidence = (P/2)*log(E_alpha) + (1/2)*log(det(sigma)) ...
        + (1/2)*w'*sigma^(-1)*w ...
        + sum(-log(1+exp(-xi)) - (1/2)*xi + (lambda'.*(xi.^2)));
    
    d_w = norm(w-wold)/(norm(wold)+1e-32);
    
    fprintf(['Iteration %i: evidence = %f, wchange = %f, '
        'rmse = %f, an = %f, bn = %f\n'], ...
        i, evidence, d_w, rmse, an, bn);
    
    i = i + 1;
end

if(i < maxit)
    fprintf('Optimization of alpha and beta successfull.\n');
else
    fprintf('Optimization terminated due to max iteration.\n');
end

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
