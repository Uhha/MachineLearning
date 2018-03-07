function varargout = sfunc(varargin)
   [varargout{1:nargout}] = feval(varargin{:});
end  
  
  function [theta] = gradientDescent(X, y, theta, alpha, num_iters)
      m = length(y); 
      for iter = 1:num_iters
         theta -= alpha*(1/m)* ((X*theta - y)' * X)';
         %J_history(iter) = computeCostMulti(X, y, theta);
      end
   
  

    function J = computeCost(X, y, theta)
      m = length(y); 
      J = (1/(2*m)) * sum(((X * theta) - y).^2);
    end
  
 
    function [theta] = gradientDescentSmart(X, y, initial_theta)
      options = optimset('GradObj', 'on', 'MaxIter', 400);
      [theta, cost] = ...
        fminunc(@(t)(computeCost(X, y, t)), initial_theta, options);
    end




