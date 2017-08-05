%% 2nd order difference matrix
% U2 is a (n-2)-by-n matrix
function U2 = diffmat2(n)

U2 = diag(ones(n-2, 1), 0) + diag(-2*ones(n-3, 1), 1) + diag(ones(n-4, 1), 2); 
U2 = [U2, zeros(n-2, 2)]; % last 2 columns
U2(end, end - 1) = -2; U2(end - 1, end - 1) = 1; 
U2(end, end) = 1; % the last column

end