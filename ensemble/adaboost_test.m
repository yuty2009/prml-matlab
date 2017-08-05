function ypred = adaboost_test(abmodel,X,ftest)

[N,P] = size(X);

K = abmodel.K;
alpha = abmodel.alpha;

y = zeros(N,1);
for i = 1:length(abmodel.models)
    y1 = ftest(abmodel.models{i}, X);
    y = y + y1.*alpha(i);
end

ypred = sign(y);