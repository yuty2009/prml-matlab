function sae = saeinit(depth,SN)

sae = mlpinit(SN);
sae.depth = depth; % number of AEs

for i = 1:sae.depth
    sae.ae{i} = mlpinit([SN(i),SN(i+1),SN(i)]);
end
