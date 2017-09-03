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

% Multiply theta to this vector so that it's not regularize theta(0):
% https://www.coursera.org/learn/machine-learning/resources/Zi29t - check after: "Note Well: The second sum," ... 
non_reg_first_parameter = ones(size(theta));
non_reg_first_parameter(1) = 0;

h = sigmoid(X * theta);
J = (1 / m) * (-y' * log(h) - (1 - y)' * log(1 - h)) + (lambda / (2 * m)) * sum(theta.^2 .* non_reg_first_parameter);

grad = ((1 / m) * sum((h - y) .* X))' .+ (lambda / m) * (theta .* non_reg_first_parameter);

% =============================================================

end
