function [w] = lsreg(y, X)
%% least square regression
% X: N by P feature matrix, N number of samples, P number of features
% y: N by 1 target vector
% b: w by 1 regression coefficients

w = inv(X'*X)*X'*y;
