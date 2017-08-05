%% Gaussian Process Regression
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% opts: options for kernel parameters
function model = gpreg(y,X,opts)

if ~isfield(opts,'beta'), opts.beta = 10; end
if ~isfield(opts,'itype'), opts.itype = 1; end % input type
if ~isfield(opts,'ktype'), opts.ktype = 'gpkernel'; end

PHI = cat(2,ones(size(X,1),1),X);

theta = [opts.args(:);opts.beta];

% [cost,grad] = gpcost(theta,y,PHI,opts);
% grad1 = numgrad(@(p)gpcost(p,y,PHI,opts),theta);
% disp([grad grad1]); 
% diff = norm(grad1-grad)/norm(grad1+grad);
% disp(diff);

options.method = 'lbfgs';
options.maxIter = 200;
options.display = 'on';
[opttheta,cost] = minFunc(@(p)gpcost(p,y,PHI,opts), ...
    theta,options);

opts.args = opttheta(1:length(opts.args));
beta = opttheta(end);

N = size(PHI,1);
if opts.itype > 0
    K = kernel(PHI',PHI',opts.ktype,opts.args);
else
    K = X;
end
invC = (K+(1/beta)*eye(N))^(-1);
alpha = invC*y;

model.sv = PHI;
model.svind = 1:N;
model.alpha = alpha;
model.b = 0;
model.beta = beta;
model.invC = invC;
model.opts = opts;

end
    
function [cost,grad] = gpcost(theta,y,X,opts)

    opts.args = theta(1:length(opts.args));
    beta = theta(end);

    N = size(X,1);
    if opts.itype > 0
        K = kernel(X',X',opts.ktype,opts.args);
    else
        K = X;
    end
    C = K+(1/beta)*eye(N);
    invC = C^(-1);
    
    cost = -0.5*(log(det(C))+y'*invC*y+N*log(2*pi));
    
    grad = zeros(length(theta),1);
    dK = kderiv(X',X',opts.ktype,opts.args);
    for i = 1:length(opts.args)
        grad(i) = -0.5*trace(invC*dK{i})+0.5*y'*invC*dK{i}*invC*y;
    end
    dbeta = -beta^-2*eye(N);
    grad(end) = -0.5*trace(invC*dbeta)+0.5*y'*invC*dbeta*invC*y;
 
end