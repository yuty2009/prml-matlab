n = 5;
A = eye(n,n); C = randn(n,n); b = 1;

cvx_begin
    variable X(n,n) symmetric;
    minimize( trace( C * X ) );
    subject to
    trace( A * X ) == b;
    X == semidefinite(n);
cvx_end