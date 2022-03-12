function dbn = dbninit(depth,SN)

dbn = mlpinit(SN);
dbn.depth = depth; % number of RBMs

for i = 1:dbn.depth
    dbn.rbm{i} = rbminit(SN(i),SN(i+1),'zero');
end
