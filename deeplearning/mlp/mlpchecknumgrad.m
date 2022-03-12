function mlpchecknumgrad(mlp,y,X)
    epsilon = 1e-4;
    er      = 1e-8;
    for L = 1:mlp.NL-1
        for i = 1:size(mlp.W{L},1)
            for j = 1 : size(mlp.W{L},2)
                mlp_m = mlp; mlp_p = mlp;
                mlp_m.W{L}(i,j) = mlp.W{L}(i,j) - epsilon;
                mlp_p.W{L}(i,j) = mlp.W{L}(i,j) + epsilon;
                mlp_m = mlpff(mlp_m,X);
                mlp_m = mlpbp(mlp_m,y);
                mlp_p = mlpff(mlp_p,X);
                mlp_p = mlpbp(mlp_p,y);
                dW = (mlp_p.cost - mlp_m.cost)/(2*epsilon);
                e = abs(dW - mlp.dW{L}(i,j));
                
                assert(e < er, 'numerical gradient checking failed');
            end
        end
    end
end