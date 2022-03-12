%% Bayes linear regression with elastic net regularization
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% b: P by 1 regression coefficients
% b0: the intercept
function [varargout] = benetreg(y, X)

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N,P] = size(PHI);

% initialize with least square estimation
if (P > N)
    invC = woodburyinv(eye(P),PHI',PHI,eye(N));
else
    invC = (eye(P)+ PHI'*PHI)^(-1);
end
w = invC*PHI'*y;
% w = ones(P,1); % rough initialization
alphas = 2*ones(P,1);
beta = 10;
rho = 1;

% stop conditions
d_w = Inf;
maxit = 500;
stopeps = 1e-6;
maxvalue = 1e9;

% [v,d] = eig(PHI'*PHI);
d = myeig(PHI);
d = diag(d);

i = 1;
while (d_w > stopeps)  && (i < maxit)
    wold = w;
    
    %% Regarding the update of the hyperparameters, refer to Page 347-348
    %% of PRML
    
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
    if (P1 > N1)
        Sigma1 = woodburyinv(diag(alphas1+rho), PHI1', PHI1, (1/beta)*eye(N));
    else
        Sigma1 = (diag(alphas1+rho) + beta*PHI1'*PHI1)^(-1);
    end
    
    diagSigma1 = diag(Sigma1);
    w1 = beta*Sigma1*PHI1'*y;
    w(index1) = w1;
    if(~isempty(index0)) w(index0) = 0; end
    
    rmse = sum((y-PHI*w).^2);

    gamma1 = 1 - (alphas1+rho).*diagSigma1;
    alphas1 = max(gamma1,eps)./(w1.^2+1e-32) - rho;
    alphas(index1) = max(alphas1,eps); % ensure alpha is positive
    rho = (P-w1'*w1-sum(diagSigma1))/(sum(1./(alphas1+rho))+1e-32);
    beta  = max(N-sum(gamma1),eps)/(rmse+1e-32);
    
    evidence = (1/2)*sum(log(alphas)) + (N/2)*log(beta) - ...
        (beta/2)*rmse - (1/2)*w'*diag(alphas)*w - ...
        (1/2)*sum(log(beta*d+alphas)) - (N/2)*log(2*pi);
    
    d_w = norm(w-wold)/(norm(wold)+1e-32);
    
    disp(['Iteration ' num2str(i)  ': evidence = ' num2str(evidence) ...
        ', wchange = ' num2str(d_w) ', rmse = ' num2str(rmse) ...
        ', rho = ' num2str(rho) ', beta = ' num2str(beta)]);
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
