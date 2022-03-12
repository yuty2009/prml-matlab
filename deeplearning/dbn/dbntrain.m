function dbn = dbntrain(dbn,y,X,opts)

XTemp = X;
for i = 1:dbn.depth
    disp(['Training RBM ' num2str(i) '/' num2str(dbn.depth)]);
    dbn.rbm{i} = rbmtrain(dbn.rbm{i},XTemp,opts);
    XTemp = rbmup(dbn.rbm{i},XTemp);
end
% unfolding dbn to mlp
dbn.TF = dbn.rbm{1}.TF;
for i = 1:dbn.depth
    dbn.W{i} = dbn.rbm{i}.W;
    dbn.b{i} = dbn.rbm{i}.b;
end

disp('Training the output layer');
switch (dbn.oTF)
    case 'linear'
        oW = ridgereg(y,XTemp,opts.lambda);
    case {'sigmoid','softmax'}
        oW = softmax(y,XTemp,opts.lambda);
end
dbn.W{dbn.depth+1} = oW(2:end,:);
dbn.b{dbn.depth+1} = oW(1,:);

% fine tune
% dbn = mlptrain(mlp,y,X,opts);
