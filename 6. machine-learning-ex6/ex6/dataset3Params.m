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

% store given parameters for C and sigma
params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30] ;

% store size of parameter array
n = size(params, 2) ;

% create an empty array to store the results
results = zeros(n * n, 3) ; 
counter = 0 ;

for c = 1:n
    for s = 1:n
        
        % keep counter as index in results matrix 
        counter = counter + 1 ;
        
        % create temp values for C and sigma to be used in model
        C_temp = params(c) ;
        sigma_temp = params(s) ;
        
        % develop the model
        model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp)); 
        
        % predict cross validation y values
        predictions = svmPredict(model, Xval) ;
        
        % calculate the prediction error
        pred_error = mean(double(predictions ~= yval)) ;
        
        % store the prediction error in column 1 of the results
        results(counter, 1) = pred_error ;
        
        % store the C parameter in column 2 of results
        results(counter, 2) = C_temp ;
        
        % store the C parameter in column 3 of results
        results(counter, 3) = sigma_temp ;
    end
end

% extract the minimum value and index from each column
[Min,Idx] = min(results(1:end, 1));

% save the minimum C and sigma
C = results(Idx, 2) ;
sigma = results(Idx, 3) ;

% =========================================================================

end
