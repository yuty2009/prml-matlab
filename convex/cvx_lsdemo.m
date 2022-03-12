m = 16; n = 8;
A = randn(m,n);
b = randn(m,1);

xx = A\b;

cvx_begin
    variable x(n);
    minimize( norm(A*x-b) );
cvx_end