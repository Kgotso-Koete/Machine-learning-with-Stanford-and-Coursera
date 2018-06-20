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

% first calculate z to be used in the sigmoid hypothesis
h = X * theta ; % product of vectors to apply theta parameters

% calculate the error term
squared_err = (h - y).^2 ;

% calculate unregularized cost
J_unregularized = (1/(2* m)) * sum(squared_err) ;

% set theta(1) to 0
theta(1) = 0 ;
 
%  calculate the sum of the squares of theta and scale by lambda
J_reg_term = (lambda / (2 * m)) * (theta' * theta) ;

% calculate final cost
J = J_unregularized + J_reg_term ;

% calculate the temporary gradient
grad_unregularized = (1 / m) * (X' * (h - y)) ; % vector product also includes the required summation

% calculate the regularized gradient term as theta scaled by (lambda / m)
grad_reg_term = theta * (lambda / m) ;

% calculate final gradient
grad = grad_unregularized + grad_reg_term ;

 
% =========================================================================

grad = grad(:);

end
