function Y = aeencode(ae,X)

Y = fvalue(X*ae.W{1}+repmat(ae.b{1},size(X,1),1),ae.TF);