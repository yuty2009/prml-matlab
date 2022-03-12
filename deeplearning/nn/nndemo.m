clc
clear

inputs = rand([5,28*28]);
outputs = rand([5,10]);
outputs=(outputs==repmat(max(outputs,[],2),1,size(outputs,2)));

lr = 0.001;
batchsize = 5;
layers{1} = layerinit([batchsize, 28*28], [28*28, 100], [1, 100], 'sigmoid');
layers{2} = layerinit([batchsize, 100], [100, 10], [1, 10], 'softmax');
net = netinit(layers, 'crossentropy');

mlp = mlpinit([28*28, 100, 10]);
mlp.W{1} = layers{1}.weights; mlp.b{1} = layers{1}.biases;
mlp.W{2} = layers{2}.weights; mlp.b{2} = layers{2}.biases;
mlp.TF = 'sigmoid';
mlp.oTF = 'softmax';
mlp.dropout = 0;

rng('default');
mlp = mlpff(mlp, inputs);
mlp = mlpbp(mlp, outputs);
mlpchecknumgrad(mlp, outputs, inputs);

[net, act, as] = netforward(net, inputs, 1);
net.cost = lvalue(act, outputs, net.loss);
delta = lderiv(act, outputs, net.loss);
net = netbackward(net, delta, as);
netcheckgrads(net, inputs, outputs);