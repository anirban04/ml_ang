function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


%computing the cost %

h = sigmoid(X * theta);
logA = log(h);
logB = log(1- h);
transposeA = y';
transposeB = (1 - y)';
termA = transposeA * logA;
termB = transposeB * logB;
thetaSquares = theta .^2;
thetaSum = sum(thetaSquares) - thetaSquares(1);
regTerm = (lambda / (2 * m)) * thetaSum;

% Note: 
% Cost with regularization = Cost w/o reg + ((lambda/2 * m) * thetaSum 
% where thetaSum is the sum of squares of theta(2) to theta(n). 
% No theta(1) squared term is included in sum.

J = ((1/m) *  (-termA - termB)) + regTerm;

% computing the gradient parameters%
% Similar to computing gradient without reg + the regTerm. 
m = size(X, 1);
grad = (X' * (h - y)) ./ m; 
regTerm = (lambda / m) .* theta;
regTerm(1) = 0;
grad = grad + regTerm;

% =============================================================

end
