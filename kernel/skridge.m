%% Sparse Kernel ridge regression
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% opts: options for kernel parameters
function model = skridge(y, X, opts)

if ~isfield(opts,'lambda'), opts.lambda = 1e-4; end
if ~isfield(opts,'tol'), opts.tol = 1e-6; end
if ~isfield(opts,'itype'), opts.itype = 1; end % input type
if ~isfield(opts,'ktype'), opts.ktype = 'linear'; end
if ~isfield(opts,'args'), opts.args = [1,0]; end

N = size(X,1);
PHI = cat(2,ones(N,1),X); % add a constant column to cope with bias
if opts.itype > 0
    K = kernel(PHI',PHI',opts.ktype,opts.args);
else
    K = X;
end

alpha = zeros(N,1);
cvx_quiet(false);
cvx_begin
    variable alpha(N,1);
    minimize norm(y-K*alpha) + opts.lambda*norm(alpha, 1);
cvx_end

index = find(abs(alpha)/norm(alpha)>opts.tol);

model.sv = PHI(index,:);
model.svind = index;
model.alpha = alpha(index);
model.b = 0;
model.opts = opts;

