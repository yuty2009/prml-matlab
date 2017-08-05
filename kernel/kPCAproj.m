%% Kernel Principle Component Analysis
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% opts: options for kernel parameters
function Y = kPCAproj(X, dim, model)

opts = model.opts;
K = kernel(X',model.sv',opts.ktype,opts.args);
Y = K*model.alpha(:,1:dim) + repmat(model.b(1:dim)',250,1);
