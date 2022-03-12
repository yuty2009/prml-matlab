% Generate discrete samples given an probability table
% @param Pz: 1 by K probability table of the K states
% @output X: M by N data points generated from the given ptable, whos
%  entries have values in [1:K]
function [X] = randptable(Pz,M,N)

if (nargin == 1)
    M = 1;
    N = 1;
end

K = length(Pz);

Pz = abs(Pz);
Pz = Pz./sum(Pz);

% calculate cdf
cPz = cumsum(Pz);

X = zeros(M,N);
for i = 1:M
    for j = 1:N
        index = 1;
        temp = rand(1);
        for k = K-1:-1:1
            if (temp > cPz(k))
                index = k + 1;
                break;
            end
        end
        X(i,j) = index;
    end
end