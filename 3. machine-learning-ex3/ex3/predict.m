function p_1 = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p_1 = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% create layer 1 by adding bias (ones) to input matrix X
a_1 = [ones(m, 1) X];

% product of vectors to apply theta parameters
z_2 = a_1 * Theta1' ; 

% calculate the hypothesis (list of probabilities that
h_2 = sigmoid(z_2) ; 

% finalize layer 2 output and add bias (ones)
a_2 = [ones(m, 1) h_2];

% ************ LAYER 2 CALCULATIONS ***********************

% product of vectors to apply theta parameters
z_3 = a_2 * Theta2' ;   

% calculate the hypothesis (list of probabilities that
h_3 = sigmoid(z_3) ; 

% use max(A, [], 2) to obtain the max and index for each row
[mx_3,idx_3] = max(h_3,[],2) ;

% save prediction index as layer 3 output
p_1 = idx_3 ;

% =========================================================================


end
