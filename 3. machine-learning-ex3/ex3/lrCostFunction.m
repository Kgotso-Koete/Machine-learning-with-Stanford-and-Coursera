function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

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

grad = grad(:); 

end
