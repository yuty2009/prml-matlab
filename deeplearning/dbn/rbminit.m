function rbm = rbminit(S1,S2,inittype)

rbm = mlpinit([S1,S2,S1],inittype);

rbm.W = rbm.W{1};
rbm.a = zeros(1,S1); % bias of visible
rbm.b = zeros(1,S2); % bias of hidden

rbm.dW = rbm.dW{1};
rbm.da = 0;
rbm.db = 0;
