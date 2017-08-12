function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
minErr = 1000000000;

cIter = 0.01;
count = 0;
a = 99999999;
while(cIter <= 30)
  sigmaIter = 0.01;
  while(sigmaIter <= 30)
  	count++;
	model= svmTrain(X, y, cIter, @(x1, x2) gaussianKernel(x1, x2, sigmaIter)); 
  	predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
	if (err < minErr)
  		C = cIter;
		sigma = sigmaIter;
		minErr = err;
	end;
	sigmaIter = sigmaIter * 3;
  end;
  cIter = cIter * 3;
end;
% =========================================================================
end
