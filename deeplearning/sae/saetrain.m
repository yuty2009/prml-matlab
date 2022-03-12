function sae = saetrain(sae,y,X,opts)

XTemp = X;
for i = 1:sae.depth
    disp(['Training AE ' num2str(i) '/' num2str(sae.depth)]);
    sae.ae{i} = mlptrain(sae.ae{i},XTemp,XTemp,opts);
    sae.ae{i}.corruption = 0; % no corruption when testing
    XTemp = aeencode(sae.ae{i},XTemp);
end
% unfolding sae to mlp
sae.TF = sae.ae{1}.TF;
for i = 1:sae.depth
    sae.W{i} = sae.ae{i}.W{1};
    sae.b{i} = sae.ae{i}.b{1};
end

disp('Training the output layer');
switch (sae.oTF)
    case 'linear'
        oW = ridgereg(y,XTemp,opts.lambda);
    case {'sigmoid','softmax'}
        oW = softmax(y,XTemp,opts.lambda);
end
sae.W{sae.depth+1} = oW(2:end,:);
sae.b{sae.depth+1} = oW(1,:);

% fine tune
% sae = mlptrain(mlp,y,X,opts);
