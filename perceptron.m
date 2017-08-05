%% Perceptron classifier
% X: N by P design matrix with N samples of M features
% t: N by 1 class labels of [-1 1] 
% w: P+1 by 1 weight vector
function [varargout] = perceptron(t,X,opts)

if nargin < 4
    opts.lrate = 1;
    opts.maxepoch = 500;
    opts.stopeps = 1e-6;
end
if ~isfield(opts,'lrate'), opts.lrate = 1; end
if ~isfield(opts,'maxepoch'), opts.maxepoch = 500; end
if ~isfield(opts,'stopeps'), opts.stopeps = 1e-6; end

PHI = cat(2, ones(size(X,1),1), X); % add a constant column to cope with bias
[N, P] = size(PHI);

w = zeros(P,1);
d_w = Inf;
error = Inf;

i = 1;
while (error > opts.stopeps)  && (i <= opts.maxepoch)
    wold = w;
    
    perm = randperm(N);
    for n = perm
        phi = PHI(n,:)';
        w = w + opts.lrate*phi*t(n);
    end
    d_w = norm(wold - w);
    y = PHI*w;
    idx = find(t~=sign(y));
    error = -sum(y(idx).*t(idx));
    
    fprintf('Epoch %d: error = %f, wchange = %f\n', i, error, d_w);
    i = i + 1;
end

disp(['Optimization terminated after ' num2str(i-1) ' epochs']);

b = w(2:P);
b0 = w(1);

if nargout == 1
    model.b = b;
    model.b0 = b0;
    varargout{1} = model;
elseif nargout == 2
    varargout{1} = b;
    varargout{2} = b0;
end