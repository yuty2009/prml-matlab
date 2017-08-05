clc
clear

datapath = 'e:\prmldata\mnist\';
mnist = mnistdata();
[train, validation, test] = mnist.load(datapath, true, false);

cnn.layers = {
    struct('type', 'input') %input layer
    struct('type', 'conv', 'outputmaps', 32, 'kernelsize', 5, 'TF', 'ReLU') %convolution layer
    struct('type', 'pool', 'kernelsize', 2, 'method', 'max') %sub sampling layer
    struct('type', 'conv', 'outputmaps', 64, 'kernelsize', 5, 'TF', 'ReLU') %convolution layer
    struct('type', 'pool', 'kernelsize', 2, 'method', 'max') %subsampling layer
    struct('type', 'dense', 'outputsize', 1024, 'TF', 'ReLU', 'dropout', true) %fully-connected dropout layer
    struct('type', 'dense', 'outputsize', 10, 'TF', 'softmax') %output layer
};
cnn.dropout = 0.75;

opts.verbose = 'on';
opts.batchsize = 50;
opts.maxepochs = 50000;
opts.reportsteps = 100;
opts.savesteps = 10000;
opts.savepath = 'models/';

inshape = size(train.images);
inshape = [inshape(2:3), inshape(1)];
outshape = size(train.labels');
cnn = cnninit(cnn, outshape, inshape);
% foo = load([opts.savepath 'mnist_checkpoint_step_20000']);
% cnn = foo.cnn;

tic;
lr_start = 5e-3;
cnn.rL = [];
for i = 1:opts.maxepochs
    [batch_X, batch_y] = train.nextbatch(opts.batchsize, true);
    batch_X = permute(batch_X, [2,3,1]);
    batch_y = batch_y';
    opts.lr = lr_start*(1+1e-4*i)^(-0.75);

    cnn = cnnff(cnn, batch_X);
    cnn = cnnbp(cnn, batch_y);
    cnn = cnnapplygrads(cnn, opts);

    if isempty(cnn.rL) cnn.rL(1) = cnn.loss; end
    cnn.rL(end + 1) = 0.00 * cnn.rL(end) + 1.00 * cnn.loss;
    
    if (mod(i, opts.savesteps) == 0)
        save([opts.savepath 'mnist_checkpoint_step_' num2str(i)], 'cnn');
    end
    
    if (mod(i, opts.reportsteps) == 0)
        [dummy, batch_ypred] = max(cnn.layers{cnn.NL}.A{1});
        [dummy, batch_ytrue] = max(batch_y);
        train_acc = mean(batch_ytrue(:) == batch_ypred(:));
        toc;
        if isfield(opts,'verbose') && strcmpi(opts.verbose,'on')
            fprintf('Step=%d/%d, lr=%.4f, loss=%.4f, training accuracy=%g\n', ...
                i, opts.maxepochs, opts.lr, cnn.loss, train_acc);
        end
        tic;
    end
end
figure; plot(cnn.rL);

y2 = test.labels';
X2 = permute(test.images, [2,3,1]);

cnn = cnnff(cnn, X2);
[dummy, ypred] = max(cnn.layers{cnn.NL}.A{1});
[dummy, ytrue] = max(y2);

acc = mean(ytrue(:) == ypred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
