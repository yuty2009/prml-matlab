function [vector] = basispoly(x, order)
    vector = zeros(length(x),order+1);
    for i=1:order+1
        vector(:,i) = x.^(i-1);
    end
end