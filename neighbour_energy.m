%% Calculate the neighbour energy of X(i,j) respect to
%% X(i,j) = -1 and X(i,j) = 1
function [E0 E1] = neighbour_energy(X,i,j)
    E_temp0 = 0;
    E_temp1 = 0;
    
    Xij0 = -1;
    Xij1 = 1;
    
    if(i == 1)
        r1 = i;
        r2 = i + 1;
    elseif(i == size(X,1))
        r1 = i - 1;
        r2 = i;
    else
        r1 = i - 1;
        r2 = i + 1;
    end

    if(j == 1)
        c1 = j;
        c2 = j + 1;
    elseif(j== size(X,2))
        c1 = j - 1;
        c2 = j;
    else
        c1 = j - 1;
        c2 = j + 1;
    end
    
    for m = r1:r2
        for n = c1:c2
            E_temp0 = E_temp0 + Xij0*X(m,n);
            E_temp1 = E_temp1 + Xij1*X(m,n);
        end
    end

    E0 = E_temp0 - Xij0*X(i,j);
    E1 = E_temp1 - Xij1*X(i,j);
end