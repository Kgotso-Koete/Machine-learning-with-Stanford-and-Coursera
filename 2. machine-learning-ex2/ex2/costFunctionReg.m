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

% first calculate z to be used in the sigmoid hypothesis
z = X * theta ; % product of vectors to apply theta parameters

% calculate the hypothesis
h = sigmoid(z) ; 

% calculate the error term
err_cost = (-y' * log(h) - (1 - y)' * log(1 - h)) ;

% calculate unregularized cost
J_unregularized = (1 / m) * (err_cost) ;

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

% =============================================================

end
