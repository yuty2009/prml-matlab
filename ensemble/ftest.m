function y = ftest(model, X)

dummy = zeros(size(X,1),1);
[y, predict_accuracy, predict_decvalue] = svmpredict(dummy, X, model);

% Y = sign(X*model.b + model.b0);