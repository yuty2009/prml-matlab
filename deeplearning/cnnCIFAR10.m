clc
clear

datapath = 'e:\prmldata\cifar-10\';
% datapath = '/Users/n0n/work/data/prmldata/cifar-10/';
train_images = [];
train_labels = [];
for i = 1:5
    load([datapath 'data_batch_' num2str(i)]);
    train_images = cat(1, train_images, data);
    train_labels = cat(1, train_labels, labels);
end
train_labels(train_labels == 0) = 10; % Remap 0 to 10
train_labels = onehot_labels(double(train_labels), 10);
inputsize = sqrt(size(train_images,2)/3)*ones(2,1);
train_images = reshape(train_images, size(train_images,1), inputsize(1), inputsize(2), 3);
train = imageset(train_images, train_labels, false);
% displayColorImageGrid(train.images(1:100,:)');

cnn.layers = {
    struct('type', 'input') %input layer
    struct('type', 'conv', 'outputmaps', 6, 'kernelsize', [5 5 3], 'TF', 'ReLU') %convolution layer
    struct('type', 'pool', 'kernelsize', [2 2 1], 'method', 'max') %sub sampling layer
    struct('type', 'conv', 'outputmaps', 12, 'kernelsize', [5 5 1], 'TF', 'ReLU') %convolution layer
    struct('type', 'pool', 'kernelsize', [2 2 1], 'method', 'max') %subsampling layer
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
inshape = [inshape(2:end), inshape(1)];
outshape = size(train.labels');
cnn = cnninit(cnn, outshape, inshape);
% foo = load([opts.savepath 'cifar10_checkpoint_step_20000']);
% cnn = foo.cnn;

tic;
lr_start = 5e-3;
cnn.rL = [];
for i = 1:opts.maxepochs
    [batch_X, batch_y] = train.nextbatch(opts.batchsize, true);
    batch_X = permute(batch_X, [2:length(inshape),1]);
    batch_y = batch_y';
    opts.lr = lr_start*(1+1e-4*i)^(-0.75);

    cnn = cnnff(cnn, batch_X);
    cnn = cnnbp(cnn, batch_y);
    cnn = cnnapplygrads(cnn, opts);

    if isempty(cnn.rL) cnn.rL(1) = cnn.loss; end
    cnn.rL(end + 1) = 0.00 * cnn.rL(end) + 1.00 * cnn.loss;
    
    if (mod(i, opts.savesteps) == 0)
        save([opts.savepath 'cifar10_checkpoint_step_' num2str(i)], 'cnn');
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

% test
load([datapath 'test_batch']);
test_images = data;
test_labels = labels;
test_labels(test_labels == 0) = 10; % Remap 0 to 10
test_labels = onehot_labels(double(test_labels), 10);
test_images = reshape(test_images, size(test_images,1), inputsize(1), inputsize(2), 3);
% test = imageset(test_images, test_labels, false);
X2 = reshape(test_images, [2:length(inshape),1]);
y2 = test_labels';

cnn = cnnff(cnn, X2);
[dummy, ypred] = max(cnn.layers{cnn.NL}.A{1});
[dummy, ytrue] = max(y2);

acc = mean(ytrue(:) == ypred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
