clc;
clear;

batch_x = rand(20, 5);
batch_y = rand(20, 2);

for output = {'softmax'} % {'linear','sigmoid','softmax'}
    disp(['output function = ' output{1}]);
    
    y = batch_y;
    if(strcmp(output,'softmax'))
        % softmax output requires a binary output vector
        y=(y==repmat(max(y,[],2),1,size(y,2)));
    end
    
    for activation_function = {'sigmoid','tanh','tanh_opt','ReLU'}
        disp(['  activation function = ' activation_function{1}]);
        
        for dropoutFraction = {0 rand()}
            disp(['    dropoutFraction = ' num2str(dropoutFraction{1})]);
            
            mlp = mlpinit([5 3 4 2]);

            mlp.TF = activation_function{1};
            mlp.oTF = output{1};
            mlp.dropout = dropoutFraction{1};

            rng('default');
            mlp = mlpff(mlp, batch_x);
            mlp = mlpbp(mlp, y);
            mlpchecknumgrad(mlp, y, batch_x);
        end
    end
end