function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1); 

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% calculate the size of X examples
m = size(X, 1) ;

% create a "distance" matrix of size (m x K) and initialize it to all zeros
distances = zeros(m, K) ;

% loop over all centroids
for i = 1:K
    
    % differences between each row in the X matrix and a centroid
    D = bsxfun(@minus, X , centroids(i,:)) ;
    
    % calculate the sum of the squares of the differences 
    result = sum(D.^2,2) ;
    
    % store column vector of the distance in distance matrix
    distances(: , i) = result ;

% save the index of the minimum distances from each exampes to each centroid
[val, loc] = min(distances') ;
idx = loc' ; 


% =============================================================

end

