function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(X(1,:));  % number of variables
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    th_change = zeros(n,1);
    diff = zeros(m,1);
    for i = 1:m % calculate individual difference for each sample
        h = theta'* X(i,:)';
        diff(i,1) = h-y(i,1);
    end
    for j = 1:n
        sum = diff' * X(:,j);
        th_change(j,1) = alpha/m*sum;
    end
    
    theta = theta - th_change;

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
