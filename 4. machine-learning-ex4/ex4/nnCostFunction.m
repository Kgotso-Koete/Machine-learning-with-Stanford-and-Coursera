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

% ****************************** PART 1: FOWRWARD FEED TO CALC COST ************************************

% LAYER 1 CALCULATIONS ***********************

% create layer 1 by adding bias (ones) to input matrix X
a_1 = [ones(m, 1) X];

% product of vectors to apply theta parameters
z_2 = a_1 * Theta1' ; 

% calculate the hypothesis (list of probabilities)
h_2 = sigmoid(z_2) ; 

% finalize layer 2 output and add bias (ones)
a_2 = [ones(m, 1) h_2];
 
% LAYER 2 CALCULATIONS ***********************

% product of vectors to apply theta parameters
z_3 = a_2 * Theta2' ;

% calculate the hypothesis (list of probabilities that
h_3 = sigmoid(z_3) ; 

% calculate rpobabilities for final layer
a_3 = h_3 ;

% Expand the 'y' output values into a matrix of single values 
eye_matrix = eye(num_labels) ;
y_matrix = eye_matrix(y,:) ;

% calculate the error term
err_cost = (-y_matrix .* log(a_3) - (1 - y_matrix) .* log(1 - (a_3))) ;
err_cost = sum(sum(err_cost)) ;

% calculate unregularized cost
J_unregularized = (1 / m) * (err_cost) ;

% set theta(1) to 0
Theta1(1:end,1) = 0 ; 
Theta2(1:end,1) = 0 ;

sum_theta_sq = sum(sum(Theta1 .* Theta1)) + sum(sum(Theta2 .* Theta2)) ;
 
%  calculate the sum of the squares of theta and scale by lambda
J_reg_term = (lambda / (2 * m)) * (sum_theta_sq) ;

% calculate final cost
J = J_unregularized + J_reg_term ;

% ****************************** PART 2: BACK PROPAGATION ************************************

% layer 3 difference
d_3 = a_3 - y_matrix  ;

% layer 2 difference (using all columns of theta except first col) 
d_2 = (d_3 * Theta2(:,2:end)) .* sigmoidGradient(z_2) ;

%  Accumulate the gradient
delta_1 = (a_1' * d_2)' ;
delta_2 = (a_2' * d_3)' ; 

%  Scale each Theta matrix by λ/m
Theta1 = (lambda / m) * Theta1 ;
Theta2 = (lambda / m) * Theta2 ;


%  Add each of these modified-and-scaled Theta matrices to the un-regularized Theta gradients
Theta1_grad = Theta1 + (delta_1 * (1 / m)) ;
Theta2_grad = Theta2 + (delta_2 * (1 / m)) ;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

  
end
