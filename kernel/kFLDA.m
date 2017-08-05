%% Kernel Fishers Linear Discriminant Analysis
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% opts: options for kernel parameters
function model = kFLDA(y, X, opts)

if ~isfield(opts,'lambda'), opts.lambda = 1e-4; end
if ~isfield(opts,'itype'), opts.itype = 1; end % input type
if ~isfield(opts,'ktype'), opts.ktype = 'linear'; end
if ~isfield(opts,'args'), opts.args = [1,0]; end

N = size(X,1);
if opts.itype > 0
    K = kernel(X',X',opts.ktype,opts.args);
else
    K = X;
end

N1 = length(find(y==1));
N2 = length(find(y==-1));
one1 = (y==1);
one2 = (y==-1);
mu1 = (1/N1)*K*one1;
mu2 = (1/N2)*K*one2;
mu = mu1 - mu2;
SIGMA = K*K' - (1/N1)*(mu1*mu1') - (1/N2)*(mu2*mu2') + opts.lambda*eye(N);

alpha = SIGMA^(-1)*mu;
b = -mean(K*alpha);

model.sv = X;
model.svind = 1:N;
model.alpha = alpha;
model.b = b;
model.opts = opts;
