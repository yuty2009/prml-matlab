%% Generate samples with a Gaussian Process distribution
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% opts: options for kernel parameters
function y = gpsample(X,opts)

if ~isfield(opts,'ktype'), opts.ktype = 'gpkernel'; end

N = size(X,1);
K = kernel(X',X',opts.ktype,opts.args);
y = mvnrnd(zeros(N,1),K,1); % zero mean and covariance K