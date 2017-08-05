%% Sequential learning (stocasitic gradient descend) for least square
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% eta: step length
% b: P by 1 regression coefficients
% b0: the intercept
function [varargout] = lsseq(y, X, eta)

PHI = cat(2, ones(size(X,1),1), X); % add a constant column to cope with bias
[N, P] = size(PHI);

w = zeros(P,1);
d_w = Inf;
rmse = Inf;
maxit = 50000;
stopeps = 0.1;

i = 1;
while (rmse > stopeps)  && (i <= maxit)
    wold = w;
    
    n = randi(N);
    phi = PHI(n,:)';
    w = w - eta*(w'*phi - y(n))*phi;
    
    d_w = norm(wold - w);
    rmse = norm(PHI*w - y);
    
    fprintf('Iteration %i, sample = %i: rmse = %f, wchange = %f\n', i, n, rmse, d_w);
    i = i + 1;
end

disp(['Optimization terminated after ' num2str(i-1) ' iterations']);

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