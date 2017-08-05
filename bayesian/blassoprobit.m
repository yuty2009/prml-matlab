%% Bayes probit regression with Laplace priors
% X: N by P feature matrix, N number of samples, P number of features
% t: N by 1 target vector [-1,1]
% b: P by 1 regression coefficients
% b0: the intercept
%
% Reference: M.A.T. Figueiredo, Adaptive sparseness for supervised
% learning, IEEE Trans Pattern Anal Mach Intell, vol. 25, (no. 9), pp.
% 1150-1159, 2003.
function [varargout] = blassoprobit(t, X)

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N,P] = size(PHI);

% initialize with least square estimation
if (P > N)
    invC = woodburyinv(1e-4*eye(P),PHI',PHI,eye(N));
else
    invC = (1e-4*eye(P)+ PHI'*PHI)^(-1);
end
w = invC*PHI'*(t-mean(t));
% w = ones(P,1); % rough initialization
alphas = 2*ones(P,1);

% stop conditions
d_w = Inf;
maxit = 500;
stopeps = 1e-6;

i = 1;
while (d_w > stopeps)  && (i < maxit)
    wold = w;
    
    % E step
    z = PHI*w;
    s = z + t.*normpdf(z)./normcdf(t.*z);
    if (P > N)
        Sigma = woodburyinv(diag(alphas), PHI', PHI, eye(N));
    else
        Sigma = (diag(alphas) + PHI'*PHI)^(-1);
    end
    w = Sigma*PHI'*s;
    
    % M step
    alphas = 1./(w.^2+1e-32); % convergence is fast but not stable
    % alphas = 1./(w.^2+diag(Sigma)+1e-32);
    
    evidence = w'*(PHI'*PHI)*w + 2*w'*PHI'*s + P;
    
    d_w = norm(w-wold)/(norm(wold)+1e-32);
    
    disp(['Iteration ' num2str(i)  ': evidence = ' num2str(evidence) ...
        ', wchange = ' num2str(d_w)]);
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
