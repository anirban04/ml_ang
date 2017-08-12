function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hTheta = X * theta;
sqErr = (1 / (2 * m)) * ((hTheta - y) .^ 2);
sumSqErr = sum(sqErr);
thetaSq = theta .^ 2;
%Do not regularize theta(0)
sumThetaSq = sum(thetaSq) - thetaSq(1);
regParam = (lambda / (2 * m)) * (sumThetaSq);
J = sumSqErr + regParam;

err1 = (hTheta - y) .* X(1);
grad(1) = (1 / m) * sum(err1);

for j=2:columns(X)
errOthers = (hTheta - y) .* X(:, j);
grad(j) = ((1 / m) * sum(errOthers)) + ((lambda / m) * theta(j));
end

% =========================================================================

grad = grad(:);

end
