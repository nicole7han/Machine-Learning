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



for i = 1:m  % for each training sample
    z = theta'* X(i,:)'; % z = theta'*x
    h = sigmoid(z);
    J = J + (-y(i,1))*log(h) - (1-y(i,1))*log(1-h);
end
J = J/m + lambda/(2*m) *  sum(theta(2:end,1).^2);


diff = zeros(m,1);
for i = 1:m  % for each training sample
    diff(i,1) = 1/(1+exp(-(theta' * X(i,:)'))) - y(i,1);
end


for j = 1:length(theta(:,1)) % for each theta
    tempx = X(:,j)';
    if j == 1
        grad(j,1) = 1/m*(tempx * diff);
    else
        grad(j,1) = 1/m*(tempx * diff) + lambda/m * theta(j,1);
    end
end




% =============================================================

end
