%% Prediction for a kernel machine
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% opts: options for kernel parameters
function y = kpredict(X, model)

if (size(X,2) == size(model.sv,2))
    PHI = X;
else
    PHI = cat(2,ones(size(X,1),1),X);
end

opts = model.opts;
if opts.itype > 0
    K = kernel(PHI',model.sv',opts.ktype,opts.args);
else
    K = X;
end
y = K*model.alpha + model.b;
