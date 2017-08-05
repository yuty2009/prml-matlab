%% Softmax regression using Newton-Raphson update
% X: N by P feature matrix, N number of samples, P number of features
% t: N by 1 class labels (t=k indicate belong to class k)
% lambda: regularization coefficient
% W: P by K regression coefficients
function [W] = softmax1(t,X,lambda)

if nargin <= 2
    lambda = 1e-4;
end

% add a constant column to cope with bias
PHI = cat(2, ones(size(X,1),1), X);
[N,P] = size(PHI);

K = length(unique(t));
T = full(sparse(1:N,t,1));

nta = 1;
maxit = 500;
stopeps = 1e-6;
wchange = Inf;

% batched gradient ascend optimization
it = 1;
W = ones(P,K);
H = zeros(P*K); % the Hessian matrix
while (wchange > stopeps) && (it < maxit)
    Wold = W;
    % predict Y
    A = PHI*W;
    Y = bsxfun(@minus, A, max(A,[],2));
    Y = exp(Y);
    Y = bsxfun(@rdivide, Y, sum(Y,2));
    % gradient
    deltaW = (1/N)*PHI'*(Y-T) + lambda*W;
    % Hessian
    II = eye(K);
    for i = 1:K
        yi = Y(:,i);
        for j = i:K
            yj = Y(:,j);
            R = diag(yi.*(II(i,j)-yj));
            Hij = lambda*eye(P) + PHI'*R*PHI;
            H((i-1)*P+(1:P),(j-1)*P+(1:P)) = Hij;
        end
    end
    % gradient descend update
    Wvec = W(:);
    deltaWvec = deltaW(:);
    Wvec = Wvec - nta*H^(-1)*deltaWvec;
    W = reshape(Wvec,P,K);
    % termination condition
    wchange = W - Wold;
    wchange = norm(wchange(:));
    fprintf('Iteration %i: change = %f\n', it, wchange);
    it = it + 1;
end
