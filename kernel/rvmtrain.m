%% Relevance Vector Machine
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% opts: options for kernel parameters
function model = rvmtrain(y, X, opts)

if ~isfield(opts,'tol'), opts.tol = 1e-9; end
if ~isfield(opts,'itype'), opts.itype = 1; end % input type
if ~isfield(opts,'ktype'), opts.ktype = 'linear'; end
if ~isfield(opts,'args'), opts.args = [1,0]; end
if ~isfield(opts,'method'), opts.method = 'bardreg'; end

if opts.itype > 0
    K = kernel(X',X',opts.ktype,opts.args);
else
    K = X;
end

switch(opts.method)
    case 'bardreg'
        [alpha,b] = bardreg(y,K);
    case 'blassoreg'
        [alpha,b] = blassoreg(y,K);
    case 'benetreg'
        [alpha,b] = benetreg(y,K);
    case 'bardlog'
        [alpha,b] = bardlog(y,K);
    case 'blassolog'
        [alpha,b] = blassolog(y,K);
    case 'blassoprobit'
        [alpha,b] = blassoprobit(y,K);
    otherwise
        disp('unknown method');
end
        
index = find(abs(alpha)/norm(alpha)>opts.tol);

model.sv = X(index,:);
model.svind = index;
model.alpha = alpha(index);
model.b = b;
model.opts = opts;
