function [X_norm, mu, sigma] = featureNormalizeExceptFirst(X)

  X_norm = X;
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));

  mu = mean(X);
  sigma =  std(X);

  mu(:,1) = 0;
  sigma(:,1) = 1;
  
  X_norm = (X - mu)./sigma;
  
end
