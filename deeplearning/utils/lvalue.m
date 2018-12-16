%% Compute the loss
function loss = lvalue(ypred, ytrue, func)

switch(func)
    case 'mse'
        loss = 0.5*(1/size(ytrue,1))*sum((ytrue(:)-ypred(:)).^2);
    case 'crossentropy'
        loss = -(1/size(ytrue,1))*ytrue(:)'*log(ypred(:));
    otherwise
        loss = 0;
        disp('unknown transfer function');
end
