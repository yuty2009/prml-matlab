%% Independent Component Analysis
%  through maximum likelihood estimation
%  X: P channel by N sample points mixed signal
%  W: P by P demixing matrix
%  S: P by N demixed signal
function [icasig,A,W] = ICAML1(X)

[P,N] = size(X);

if P > N
    disp('Too few sample points.');
    return;
end

mX = mean(X,2);
X = X - repmat(mX,1,N);
fprintf('Training data range: %g to %g\n',min(min(X)),max(max(X)));

[X,sphere] = whiten(X);

maxit = 5000;
gamma = 10^(-2); % learning rate
stopeps = 10^(-6);
wchange = Inf;

% batched gradient ascend optimization
it = 1;
W = randn(P);
while (wchange > stopeps) && (it < maxit)
    Wold = W;
    deltaW = X*(1-2*sigmoid(X'*W)) + (W')^(-1);
    W = Wold + gamma*deltaW;
    
    wchange = W - Wold;
    wchange = norm(wchange(:));
    
    fprintf('Iteration %i: change = %f\n', it, wchange);
    it = it + 1;
end

icasig = W*(X+repmat(sphere*mX,1,N));
A = (W*sphere)^(-1);
