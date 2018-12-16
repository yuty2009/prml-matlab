%% Output delta of a specific loss function
function delta = lderiv(ypred, ytrue, func)

switch(func)
    case 'mse'
        delta = ypred - ytrue;
    case 'crossentropy'
        delta = ypred - ytrue;
    otherwise
        disp('unknown transfer function');
end
