%% Evaluates kernel function.
% where k: a x b -> R is a kernel function given by identifier type 
% and argument arg:
% Identifier    Name           Definition
% 'linear'  ... linear kernel  k(a,b) = a'*b
% 'poly'    ... polynomial     k(a,b) = (a'*b+arg[2])^arg[1]
% 'rbf'     ... RBF (Gaussian) k(a,b) = exp(-0.5*||a-b||^2/arg[1]^2)
% 'sigmoid' ... Sigmoidal      k(a,b) = tanh(arg[1]*(a'*b)+arg[2])
function K = kkernel(X1, X2, type, args)

    N1 = size(X1,2);
    N2 = size(X2,2);
    K = zeros(N1,N2);

    if size(X1,1) ~= size(X2,1)
        error('kernel of two variables with different dim');
    end
    
    for i = 1:N1
        xi = X1(:,i);
        for j = 1:N2
            xj = X2(:,j);
            switch(type)
                case 'linear'
                    K(i,j) = xi'*xj;
                case 'poly'
                    K(i,j) = (xi'*xj+args(2))^args(1);
                case {'gaussian','rbf'}
                    K(i,j) = exp(-(xi-xj)'*(xi-xj)/(2*args(1)^2));
                case {'sigmoid','tanh'}
                    K(i,j) = tanh(args(1)*xi'*xj+args(2));
                case 'gpkernel'
                    K(i,j) = args(1)*exp(-(args(2)/2)*(xi-xj)'*(xi-xj)) ...
                        + args(3) + args(4)*xi'*xj;
                otherwise
                    error('unknown kernel type');
            end
        end
    end
end