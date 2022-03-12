clc
clear;

% O��Q �ֱ�Ϊ�۲�״̬����HMM״̬��
O = 3;
Q = 3;
% ���ڲ��Ե����ݼ�, ��TMD��׼ȷѽ,����,���˵�һ������
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
% loglik ������data�����hmm(������Ϊprior2, transmat2, obsmat2)��ƥ��ֵ��Խ��˵��Խƥ�䣬0Ϊ����ֵ��
% pathΪviterbi�㷨�Ľ������������path
B = multinomial_prob(data,obsmat2);
path = viterbi_path(prior2, transmat2, B)