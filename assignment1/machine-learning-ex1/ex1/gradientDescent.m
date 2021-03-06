function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    predictions = X * theta;
    err = (predictions - y);
    temp1 = err .* X(:, 1);
    temp2 = err .* X(:, 2);
    sum1 = sum(temp1);
    sum2 = sum(temp2);
    delta1 = sum1 / m;
    delta2 = sum2 / m;

    theta(1) =  theta(1) - (alpha * delta1);
    theta(2) =  theta(2) - (alpha * delta2);

    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
