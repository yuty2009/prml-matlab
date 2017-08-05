% Compute numerical gradient for cost function J with respect to theta
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
function grad = numgrad(J,theta)

EPSILON = 1e-4;

% Initialize numgrad with zeros
grad = zeros(size(theta));

for i = 1:length(theta)
    e = zeros(size(theta));
    e(i) = 1;
    grad(i) = (J(theta+e*EPSILON)-J(theta-e*EPSILON))/(2*EPSILON);
end