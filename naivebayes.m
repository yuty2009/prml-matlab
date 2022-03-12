%% Naive Bayesian classifier
% X: N by P feature matrix, N number of samples, P number of features
%    including both binary (0 or 1) features (Bernoulli Naive Bayesian)
%    and real-value features (Gaussian Naive Bayesian) condition
% y: N by 1 target vector in {1,...,K}
% model: the trained model
% prior: the class prior probabilities
% theta: the class conditional feature probability table, i.e., theta(k,p)
%      denotes the probability of p(x(p)=1|y=k)
% To make a prediction for a new Xnew, calculate ynew = Xnew*theta'
function [model] = naivebayes(y,X,inputtype)

if nargin < 3
    inputtype = 'binary';
end
model.inputtype = inputtype;
model.class_labels = sort(unique(y));

[N,P] = size(X);
K = length(unique(y));
theta = zeros(K,P);
NCs = zeros(K,1);
for i = 1:K
  idx = find(y==model.class_labels(i));
  X1 = X(idx,:);
  NCs(i) = length(idx);
  switch(inputtype)
      case 'binary'
          theta(i,:) = mean(X1);
      case 'gauss'
          mu(i,:) = mean(X1);
          sigma(i,:) = var(X1) + eps;
  end
end

model.prior = (NCs+1)/sum(NCs+1); % Laplace smoothing
switch(inputtype)
  case 'binary'
      model.theta = theta;
  case 'gauss'
      model.mu = mu;
      model.sigma = sigma;
end
