  
    function [theta] = gradientDescentSmart(X, y, initial_theta)
      options = optimset('GradObj', 'on', 'MaxIter', 400);
      [theta, cost] = ...
	      fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
    end

    function [J, grad] = costFunction(theta, X, y)
      m = length(y); % number of training examples
      J = (1/(2*m)) * sum(((X * theta) - y).^2);
      grad = 1/m * ((X*theta) - y)' * (X );

    end