function abmodel = adaboost_train(y, X, ftrain, ftest, maxit)

[N,P] = size(X);
K = length(unique(y));

w = ones(N,1)/N; % weights of samples
for i = 1:maxit
    
    randnum = rand(1,N);
    cumw = cumsum(w);
    indices = zeros(1,N);
    
    for j = 1:N
        % Find which bin the random number falls into
        loc = max(find(randnum(j) > cumw)) + 1;
        if isempty(loc)
            indices(j) = 1;
        else
            indices(j) = loc;
        end
    end
    
    % and now train the classifier
    models{i} = ftrain(y(indices),X(indices,:));
    ypred = ftest(models{i},X);
    
    % Ek <- Training error of Ck
    E(i) = sum(w.*(ypred ~= y));
    % E(i) = sum((PY ~= y))/length(PY);
    
    % alpha(i) = 1/2*ln(1-E(i))/E(i))
    alpha(i) = 0.5*log((1-E(i))/E(i));
    
    if (E(i) == 0)
        disp('Finish training with zero error');
        break;
    end
    
    % w(i+1) = w(i)/Z*exp(+/-alpha(i))
    w  = w.*exp(-alpha(i)*ypred.*y);
    w  = w./sum(w);
end

abmodel.models = models;
abmodel.K = K;
abmodel.E = E;
abmodel.alpha = alpha;