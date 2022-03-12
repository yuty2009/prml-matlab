%% Calculate the Woodbury identity
% (A + BD^{-1}C)^{-1} = A^{-1} - A^{-1}B(D+CA^{-1}B)^{-1}CA^{-1}
% which is useful when A is large and diagonal, and hence easy to invert,
% while B has many rows but few columns (and conversely for C) so that
% the right-hand side is much cheaper to evaluate than the left-hand side.
% We consider this condition only.
function WD = woodburyinv(A,B,C,D)

diagA = diag(A);
invA = 1./diagA;
% WD = invA - invA*B*(D+C*invA*B)^(-1)*C*invA;

n = size(invA,1);
aa = sparse(1:n,1:n, invA)*B;
bb = (D+C*aa)^(-1);
cc = C*sparse(1:n,1:n, invA);
WD = sparse(1:n,1:n, invA) - aa*bb*cc;
