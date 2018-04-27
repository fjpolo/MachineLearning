function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.






% =========================================================================
%range  = max(X(:,2)) - min(X(:,2));
%mean = 0;
%for i = 1:97
%  mean = mean + X(i,2);
%endfor
%mean = mean/97;
% Normalize vector
%for i = 1:97
%  X(i,2) = (X(i,2)-mean)/range;
%endfor
% The hypothesis (also called the prediction) is simply the product of X and theta.
h = X*theta;
% The second line of code will compute the difference between the hypothesis and 
%y - that's the error for each training example. 
error = h - y;
% he third line of code will compute the square of each of those error terms 
%(using element-wise exponentiation)
error_sqr = error.^2;
% Now, we'll finish the last two steps all in one line of code. You need to 
%compute the sum of the error_sqr vector, and scale the result (multiply) by 
%1/(2*m). That completed sum is the cost value J.
%J = (1/(2*97))*sum(error_sqr);
m = size(X,1);
J = (1/(2*m))*sum(((X*theta)-y).^2);
%
end
