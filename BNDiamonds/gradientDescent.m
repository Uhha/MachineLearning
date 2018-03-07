  function [theta] = gradientDescent(X, y, theta, alpha, num_iters)
      m = length(y); 
      for iter = 1:num_iters
         theta -= alpha*(1/m)* ((X*theta - y)' * X)';
         %J_history(iter) = computeCostMulti(X, y, theta);
      end
   