%% Naive Bayesian classifier
% X: N by P feature matrix, N number of samples, P number of features
%     we assume binary (0 or 1) features here (Bernoulli Naive Bayes), 
%     since real-value features condition is processed with (linear or
%     quadratic) discriminant analysis
% model: the trained model
% y: N by 1 target vector in {1,...,K}
% py: N by K target probabilities, py(n,k) = p(y=k|x(n),params)
function [varargout] = naivebayespredict(X,model)

[N,P] = size(X);
K = length(model.prior);
logprior = log(model.prior+eps);

Y = zeros(N,K);
switch(model.inputtype)
    case 'binary'
        theta = model.theta;
        X0 = not(X);
        logT1 = log(theta+eps);
        logT0 = log(1-theta+eps);
        for i=1:K
            L1 = bsxfun(@times,logT1(i,:),X);
            L0 = bsxfun(@times,logT0(i,:),X0);
            Y(:,i) = sum(L0+L1,2) + logprior(i);
        end
    case 'gauss'
        mu = model.mu;
        sigma = model.sigma;
        for i=1:K
            X1 = X - repmat(mu(i,:),N,1);
            invSigma = diag(1./sigma(i,:));
            Y(:,i) = -0.5*diag(X1*invSigma*X1') - 0.5*sum(log(sigma(i,:))) + logprior(i);
        end
end

Ytemp = bsxfun(@minus,Y,max(Y,[],2));
Ytemp = exp(Ytemp);
py = bsxfun(@rdivide,Ytemp,sum(Ytemp,2));

[dummy,y] = max(py,[],2);
for i = 1:length(y)
    y(i) = model.class_labels(y(i));
end

varargout{1} = y;
if nargout > 1
    varargout{2} = py;
end
