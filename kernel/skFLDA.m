%% Sparse Kernel Fishers Linear Discriminant Analysis
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% opts: options for kernel parameters
function model = skFLDA(y, X, opts)

if ~isfield(opts,'lambda'), opts.lambda = 1e-4; end
if ~isfield(opts,'tol'), opts.tol = 1e-6; end
if ~isfield(opts,'itype'), opts.itype = 1; end % input type
if ~isfield(opts,'ktype'), opts.ktype = 'linear'; end
if ~isfield(opts,'args'), opts.args = [1,0]; end

N = size(X,1);
if opts.itype > 0
    K = kernel(X',X',opts.ktype,opts.args);
else
    K = X;
end

b = 0;
alpha = zeros(N,1);
cvx_quiet(false);
cvx_begin
    variable b;
    variable alpha(N,1);
    minimize norm(y-K*alpha-b) + opts.lambda*norm(alpha, 1);
cvx_end

index = find(abs(alpha)/norm(alpha)>opts.tol);

model.sv = X(index,:);
model.svind = index;
model.alpha = alpha(index);
model.b = b;
model.opts = opts;
