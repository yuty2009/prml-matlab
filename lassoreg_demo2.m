 
clc
clear

beta = [3,1.5,0,0,2,0,0,0]';
P = length(beta);
sigma = 3;
rho = 0.5;
Sigma = zeros(P);
for i = 1:P
    for j = 1:P
        Sigma(i,j) = rho^abs(i-j);
    end
end

N = 20;
X = mvnrnd(zeros(P,1),Sigma,N);
% X = svmscale(X,[-1,1],'range','s');
y = X*beta + sigma*randn(N,1);

lambda = 1e-1;
methods = {'ridge','lasso','bard','blasso','benet'};
M = length(methods);
ws = zeros(P,M);
for i = 1:M
    disp(methods{i});
    switch(methods{i})
        case 'ridge'
            [w,b] = ridgereg(y, X, lambda);
        case 'lasso'
            [w,b] = lassoreg(y, X, lambda);
        case 'bard'
            [w,b] = bardreg(y, X);
        case 'blasso'
            [w,b] = blassoreg(y, X);
        case 'benet'
            [w,b] = benetreg(y, X);
    end
    ws(:,i) = w;
end

%% visualization
figure;
subplot(M+1,1,1); bar(beta); title('true');
for i = 1:M
    subplot(M+1,1,i+1); bar(ws(:,i)); title(methods{i});
end
