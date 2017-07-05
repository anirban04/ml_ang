function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
accumulate = 0;
for i=1:m
	l1 = X(i, :);
	l1 = [1, l1];
	l2 = sigmoid(l1 * Theta1');
	l2 = [1, l2];
	l3 = sigmoid(l2 * Theta2');

	hThetaXk = l3;
	% Generate a column vector if the form
	% 0
	% 0
	% 0
	% 0
	% 1 indicates that y for this training example is 5  
	% 0
	% 0
	% ...
	yk = zeros(num_labels, 1);
	yk(y(i)) = 1;
	logA = log(hThetaXk);
	logB = log(1 - hThetaXk);
	termA = logA * yk;
	termB = logB * (1 - yk);
	temp = -(termA + termB); 
	accumulate = accumulate + temp;
end

%This is the cost function without the regularization term
J = 1/m * (accumulate);


%Now we calculate the regularization term
Theta1Sq = Theta1 .^ 2; 
Theta2Sq = Theta2 .^ 2; 

%sum all the elements of theta1 except the bias terms i.e. column1
tempTheta1 = sum(Theta1Sq);
tempTheta1(1) = 0;
sumTheta1sq = sum(tempTheta1);
%sum all the elements of theta2 except the bias terms i.e. column2
tempTheta2 = sum(Theta2Sq);
tempTheta2(1) = 0;
sumTheta2sq = sum(tempTheta2);

regTerm = (lambda/(2 * m)) * (sumTheta1sq + sumTheta2sq);

%This is the cost function with regularization
J = J + regTerm;
grad_accumulate1 = zeros(size(Theta1));
grad_accumulate2 = zeros(size(Theta2));

%implement backpropogation using algo outlines in page 9 of the excercise pdf
for i=1:m
	

	%start Step1%
	t1 = X(i, :);
	t1 = [1, t1];

	%convert from a row to a col vector
	a1 = t1';
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1; a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	%end Step1%
	


	%start Step2%
	% Generate a column vector if the form
	% 0
	% 0
	% 0
	% 0
	% 1 indicates that y for this training example is 5  
	% 0
	% 0
	% ...
	yk = zeros(num_labels, 1);
	yk(y(i)) = 1;
	delta3 = a3 - yk;
	%end Step2%



	%start Step3%
	temp = Theta2' * delta3;
	%Add a first row to be able to match size of temp, to .* correctly.
	% We will anyway throw away delta2(1) later, so this(hack) is ok.
	z2 = [1; z2];
	delta2 = temp .* sigmoidGradient(z2);
	%end Step3%



	%start Step4%
	%skip the 0th element
	delta2 = delta2(2:end);
	%accumulate the gradients over all the training examples
	grad_accumulate2 = grad_accumulate2 + (delta3 * a2');
	grad_accumulate1 = grad_accumulate1 + (delta2 * a1');
	%end Step4%
end

%start Step5%
Theta1_grad = (1 / m) * grad_accumulate1;
Theta2_grad = (1 / m) * grad_accumulate2;


% Set up matrices regTerm1 and regTerm2 such that their first columns are
% all zeroes and the second column onwards are (lambda / m) * ThetaN.
% This is done to add Regularization to the backpropogation algorithm.
regTerm1_1 = zeros(rows(Theta1), 1);
regTerm1_2 = (lambda / m) * Theta1(:, 2:end);

regTerm2_1 = zeros(rows(Theta2), 1);
regTerm2_2 = (lambda / m) * Theta2(:, 2:end);

regTerm1 = [regTerm1_1, regTerm1_2];
regTerm2 = [regTerm2_1, regTerm2_2];

Theta1_grad = Theta1_grad + regTerm1;
Theta2_grad = Theta2_grad + regTerm2;
%end Step5%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
