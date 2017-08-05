function model = ftrain(y, X)

svmoption = '-s 0 -t 0 -c 1 -g 0.00154';
model = svmtrain(y, X, svmoption);

% model = FLDA(y, X);