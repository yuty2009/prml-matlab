%% Calculate the neighbour energy of X(i,j) respect to
%% X(i,j) = -1 and X(i,j) = 1
function [Nb len] = neighbors(X,i,j)

    Nb = [];
    
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
    
    index = 0;
    for m = r1:r2
        for n = c1:c2
            if (m == i && n == j)
            else
                index = index + 1;
                Nb(index) = X(m,n);
            end
        end
    end
    len = length(Nb);
    
end