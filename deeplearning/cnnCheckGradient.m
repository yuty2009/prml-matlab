clc
clear

batch_x = rand(28,28,5);
batch_y = rand(10,5);

cnn.layers = {
    struct('type', 'input') %input layer
    struct('type', 'conv', 'outputmaps', 2, 'kernelsize', 5, 'method', 'valid') %convolution layer
    struct('type', 'pool', 'kernelsize', 2, 'method', 'max') %sub sampling layer
    struct('type', 'conv', 'outputmaps', 2, 'kernelsize', 5, 'method', 'valid') %convolution layer
    struct('type', 'pool', 'kernelsize', 2, 'method', 'max') %subsampling layer
    struct('type', 'dense', 'outputsize', 20, 'TF', 'ReLU') %fully-connected dropout layer
    struct('type', 'dense', 'outputsize', 10, 'TF', 'softmax') %output layer
};

cnn = cnninit(cnn, size(batch_y), size(batch_x));

for output = {'softmax','sigmoid','linear'}
    disp(['output function = ' output{1}]);
    
    y = batch_y;
    if(strcmp(output,'softmax'))
        % softmax output requires a binary output vector
        y=(y==repmat(max(y),size(y,1),1));
    end
    
    for activation_function = {'sigmoid','tanh','tanh_opt','ReLU'}
        disp(['  activation function = ' activation_function{1}]);
        
        for dropoutFraction = {0}
            disp(['    dropoutFraction = ' num2str(dropoutFraction{1})]);

            cnn.layers{2}.TF = activation_function{1};
            cnn.layers{4}.TF = activation_function{1};
            cnn.layers{6}.dropout = dropoutFraction{1};
            cnn.layers{cnn.NL}.TF = output{1};

            rng('default');
            cnn = cnnff(cnn, batch_x);
            cnn = cnnbp(cnn, y);
            cnnnumgradcheck(cnn, y, batch_x);
        end
    end
end