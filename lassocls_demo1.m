
clc
clear

%% generate train dataset
% designed signal
N1 = 128; % number of samples in class 1
N2 = 128; % number of samples in class 2
N = N1 + N2;
P = 256; % feature dimension
X1 = randn(N1,P);
X2 = randn(N2,P) - 10;
XRaw = cat(1, X1, X2);
% designed weights
NA = 10; % active weights
w0 = [randn(NA,1); zeros(P-NA,1)];
% design matrix and class label
sigma = 0.2;
X0 = XRaw*diag(w0) + sigma*randn(N,P);
y0 = [ones(N1,1); -1*ones(N2,1)];
% permutate the samples
perm = randperm(N);
X = X0(perm,:);
y = y0(perm,1);
% calculate the discriminability of each feature
rr = rsquare(y,X);

%%
lambda = 1e-3;
methods = {'logistic','lassolog','bardlog','blassolog'};
M = length(methods);
ws = zeros(P,M);
for i = 1:M
    disp(methods{i});
    switch(methods{i})
        case 'logistic'
            [w,b] = logistic(y, X, lambda);
        case 'lassolog'
            [w,b] = lassolog(y, X, lambda);
        case 'bardlog'
            [w,b] = bardlog(y, X);
        case 'blassolog'
            [w,b] = blassolog(y, X);
    end
    ws(:,i) = w;
end

%% visualize
figure;
subplot(M+2,1,1);
plot(w0);
axis tight;
ylim([-2 2]);
title('raw');
subplot(M+2,1,2);
plot(rr);
axis tight;
ylim([0 1]);
title('r^2');
for m = 1:M
    subplot(M+2,1,m+2);
    plot(ws(:,m));
    axis tight;
    % ylim([-2 2]);
    title(methods{m});
end