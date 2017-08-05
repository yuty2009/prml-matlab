%% Kernel Principle Component Analysis
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% opts: options for kernel parameters
function model = kPCA(X, opts)

if ~isfield(opts,'itype'), opts.itype = 1; end % input type
if ~isfield(opts,'ktype'), opts.ktype = 'linear'; end
if ~isfield(opts,'args'), opts.args = [1,0]; end

N = size(X,1);
if opts.itype > 0
    K = kernel(X',X',opts.ktype,opts.args);
else
    K = X;
end
J = ones(N)/N;
% centering the data in non-linear feature space
Khat = K - J*K - K*J + J*K*J;

[PC,D] = eig(Khat);
diagD = real(diag(D));

for i = 1:N
    if diagD(i) ~= 0
        PC(:,i) = PC(:,i)/sqrt(diagD(i));
    end
end

[diagD,ordered] = sort(-diagD);
diagD = -diagD;
alpha = PC(:,ordered);

% (implicit) de-centering back for the projected data
alpha = (eye(N)-J)*alpha;
J1 = ones(N,1)/N;
b = alpha'*(J'*K*J1-K*J1);

model.sv = X;
model.svind = 1:N;
model.alpha = alpha;
model.b = b;
model.opts = opts;
