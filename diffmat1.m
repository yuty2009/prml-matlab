%% 1st order difference matrix
% D is a (n-1)-by-n matrix
function U1 = diffmat1(n)

U1 = diag(-ones(n-1, 1), 0) + diag(ones(n-2, 1), 1);
U1 = [U1, zeros(n-1, 1)]; % last column
U1(end, end) = 1; % last column

end