%% Variational linear regression with Gamma hyperprior to promote sparsity
% (Page 486-490 of PRML)
% Soluted through variational approximation & EM.
% 
% X: N by P design matrix with N samples of M features
% t: N by 1 target values 
% w: P+1 by 1 weight vector
function [varargout] = bvarsgroupreg(y,X,group)

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N, P] = size(PHI);

if (numel(group) == 1) group = ceil([1:size(X,2)]/group); end
group = [0;group(:)]; % account for bias
groupid = unique(group);
NG = length(groupid);

% initialize with least square estimation
if (P > N)
    invC = woodburyinv(1e-4*eye(P),PHI',PHI,eye(N));
else
    invC = (1e-4*eye(P)+ PHI'*PHI)^(-1);
end
w = invC*PHI'*y;
% w = ones(P,1); % rough initialization
an = 1e-9*ones(NG,1);
bn = 1e-9*ones(NG,1);
beta = 10;

% stop conditions
d_w = Inf;
maxit = 500;
stopeps = 1e-6;
maxvalue = 1e9;

i = 1;
while (d_w > stopeps)  && (i < maxit)
    wold = w;
    
    % E step
    alphas = zeros(P,1);
    for g = 1:NG
        index_ig = find(group == groupid(g));
        alphas(index_ig) = an(g)/bn(g);
    end
    
    % eliminate very large alphas to avoid precision problem of sigma
    index0 = find(alphas > maxvalue);
    index1 = setdiff(1:P, index0);
    if (length(index1) <= 0)
        disp('Optimization terminated due that all alphas are large.');
        break;
    end
    alphas1 = alphas(index1);
    PHI1 = PHI(:,index1);
    
    [N1,P1] = size(PHI1);
    if (P1>N1)
        Sigma1 = woodburyinv(diag(alphas1), PHI1', PHI1, (1/beta)*eye(N));
    else
        Sigma1 = (diag(alphas1) + beta*PHI1'*PHI1)^(-1);
    end
    
    diagSigma1 = diag(Sigma1);
    diagSigma = zeros(P,1);
    diagSigma(index1) = diagSigma1;
    w1 = beta*Sigma1*PHI1'*y;
    w(index1) = w1;
    if(~isempty(index0)) w(index0) = 0; end
    
    % M step
    for g = 1:NG
        index_ig = find(group == groupid(g));
        w_ig = w(index_ig);
        diagSigma_ig = diagSigma(index_ig);
        
        if norm(w_ig) == 0, continue; end
        
        an(g) = an(g) + 0.5*length(index_ig);
        bn(g) = bn(g) + 0.5*(sum(diagSigma_ig)+w_ig'*w_ig);
    end
    beta = N/((y-PHI*w)'*(y-PHI*w));
    
    rmse = (y-PHI*w)'*(y-PHI*w);
    % L = (P/2)*log(alphas) + (1/2)*log(det(Sigma));
    
    d_w = norm(w-wold);
    
    fprintf(['Iteration %i: wchange = %f, ' ...
        'rmse = %f, an = %f, bn = %f\n'], ...
        i, d_w, rmse, norm(an), norm(bn));
    
    i = i + 1;
end

if(i < maxit)
    fprintf('Optimization successfull.\n');
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
