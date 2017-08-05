clc
clear

aa = rand(4, 10);
bb = rand(4, 1);

[N M] = size(aa);

rank(aa)
cvx_begin
    variable w(M,1);
    minimize norm(w, 1);
    subject to
        aa*w == bb ;
cvx_end

% minusing the row mean will cause rank deficiency
% which would make cvx failed to solve this problem
cc = aa - repmat(mean(aa), size(aa, 1), 1);

rank(cc)
cvx_begin
    variable w(M,1);
    minimize norm(w, 1);
    subject to
        cc*w == bb ;
cvx_end

dd = cc([1 3 4], :);
ee = bb([1 3 4], :);

rank(cc)
cvx_begin
    variable w(M,1);
    minimize norm(w, 1);
    subject to
        dd*w == ee ;
cvx_end