clc
clear;

% O，Q 分别为观察状态数和HMM状态数
O = 3;
Q = 3;
% 用于测试的数据集, 真TMD的准确呀,哈哈,极端的一个例子
data = [1,2,3,1,2,3,1,2,3,1;
   1,2,3,1,2,3,1,2,3,1;
   1,2,3,1,2,3,1,2,3,1;
   1,2,3,1,2,3,1,2,3,1;
   1,2,3,1,2,3,1,2,3,1;
   1,2,3,1,2,3,1,2,3,1;
   1,2,3,1,2,3,1,2,3,1;
   1,2,3,1,2,3,1,2,3,1;
   1,2,3,1,2,3,1,2,3,1;
   1,2,3,1,2,3,1,2,3,1;];

% initial guess of parameters
prior1 = normalise(rand(Q,1));
transmat1 = mk_stochastic(rand(Q,Q));
obsmat1 = mk_stochastic(rand(Q,O));

% improve guess of parameters using EM
[LL, prior2, transmat2, obsmat2] = dhmm_em(data, prior1, transmat1, obsmat1, 'max_iter', size(data,1))
% use model to compute log likelihood
data1=[1,2,3,1,2,3,1,2,3,1]
loglik = dhmm_logprob(data, prior2, transmat2, obsmat2)
% log lik is slightly different than LL(end), since it is computed after the final M step
% loglik 代表着data和这个hmm(三参数为prior2, transmat2, obsmat2)的匹配值，越大说明越匹配，0为极大值。
% path为viterbi算法的结果，即最大概率path
B = multinomial_prob(data,obsmat2);
path = viterbi_path(prior2, transmat2, B)