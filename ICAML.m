%% Independent Component Analysis
%  through maximum likelihood estimation
%  X: N channels by P sample points (observations) mixed signal
%  W: N by N demixing matrix
%  S: N by P demixed signal
%  Note that the rows of W are the spatial decomposition weights while the
%  columns of A are the mixing coefficients (scalp distribution).
function [icasig,A,W] = ICAML(X)

[N,P] = size(X);

if N > P
    disp('Too few sample points.');
    return;
end

% substract mean 
mX = mean(X,2);
X = X - repmat(mX,1,P);
fprintf('Training data range: %g to %g\n',min(min(X)),max(max(X)));
% whitten the data
[X,sphere] = whiten(X);

% random initialization
W = randn(N);

% vectorization for gradient descend
theta = W(:);

[cost,grad] = funcCost(theta,X);
grad1 = numgrad(@(N)funcCost(N,X),theta);
disp([grad grad1]);
diff = norm(grad1-grad)/norm(grad1+grad);
disp(diff);

options.maxIter = 200;
options.display = 'on';

[opttheta,cost] = minFunc(@(N)funcCost(N,X),theta,options);

W = reshape(opttheta,N,N);

icasig = W*(X+repmat(sphere*mX,1,P));
A = (W*sphere)^(-1);

end

function [cost,grad] = funcCost(theta,X)
    [N,P] = size(X);
    W = reshape(theta,N,N);
    
    dW = -(1/P)*X*(1-2*sigmoid(W*X)') - (W')^(-1);
    grad = dW(:);
    
    item1 = log((1-sigmoid(W*X)).*sigmoid(W*X));
    cost = -(1/P)*sum(item1(:)) - log(det(W));
end


