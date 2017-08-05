%% Evaluates derivatives of kernel function wrt. the args.
% where k: a x b -> R is a kernel function given by identifier type 
% and argument arg:
% Identifier    Name           Definition
% 'linear'  ... linear kernel  k(a,b) = a'*b
% 'poly'    ... polynomial     k(a,b) = (a'*b+arg[2])^arg[1]
% 'rbf'     ... RBF (Gaussian) k(a,b) = exp(-0.5*||a-b||^2/arg[1]^2)
% 'sigmoid' ... Sigmoidal      k(a,b) = tanh(arg[1]*(a'*b)+arg[2])
function dK = kkderiv(X1, X2, type, args)

    N1 = size(X1,2);
    N2 = size(X2,2);
    dK = cell(length(args),1);

    if size(X1,1) ~= size(X2,1)
        error('kernel of two variables with different dim');
    end
    
    for i = 1:N1
        xi = X1(:,i);
        for j = 1:N2
            xj = X2(:,j);
            switch(type)
                case 'poly'
                    dK{1}(i,j) = log(xi'*xj+args(2))*(xi'*xj+args(2))^args(1);
                    dK{2}(i,j) = args(1)*(xi'*xj+args(2))^(args(1)-1);
                case {'gaussian','rbf'}
                    dK{1}(i,j) = args(1)^-3*(xi-xj)'*(xi-xj) ...
                        *exp(-(xi-xj)'*(xi-xj)/(2*args(1)^2));
                case {'sigmoid','tanh'}
                    dK{1}(i,j) = (1-tanh(args(1)*xi'*xj+args(2))^2)*(xi'*xj);
                    dK{2}(i,j) = 1-tanh(args(1)*xi'*xj+args(2))^2;
                case 'gpkernel'
                    dK{1}(i,j) = exp(-(args(2)/2)*(xi-xj)'*(xi-xj));
                    dK{2}(i,j) = -args(1)*(1/2)*(xi-xj)'*(xi-xj) ...
                        *exp(-(args(2)/2)*(xi-xj)'*(xi-xj));
                    dK{3}(i,j) = 1;
                    dK{4}(i,j) = xi'*xj;
                otherwise
                    error('unknown kernel type');
            end
        end
    end
end