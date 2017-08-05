%% Kernel ridge regression
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% opts: options for kernel parameters
function model = kridge(y, X, opts)

if ~isfield(opts,'lambda'), opts.lambda = 1e-4; end
if ~isfield(opts,'itype'), opts.itype = 1; end % input type
if ~isfield(opts,'ktype'), opts.ktype = 'linear'; end
if ~isfield(opts,'args'), opts.args = [1,0]; end

N = size(X,1);
PHI = cat(2,ones(N,1),X);
if opts.itype > 0
    K = kernel(PHI',PHI',opts.ktype,opts.args);
else
    K = X;
end

alpha = (K+opts.lambda*eye(N))^(-1)*y;

model.sv = PHI;
model.svind = 1:N;
model.alpha = alpha;
model.b = 0;
model.opts = opts;
