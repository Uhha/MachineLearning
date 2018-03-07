%clear;
%data = load('BNDiamonds.txt');


y = data(:, 1);
m = length(y); % number of training examples

X =  [ones(m, 1), data(:, 2:5)]; 
X(:,2) = X(:,2).^2;
X(:,3) = X(:,3).^2;

[X mu sigma] = featureNormalizeExceptFirst(X);



theta = [0;0;0;0;0];

m = length(y);
J = (1/(2*m)) * sum(((X * theta) - y).^2);
fprintf('J with theta 0 = %f\n', J);

%thetaNQ = pinv(X'*X)*X'*y;

thetaGD = gradientDescentSmart(X, y, theta);


%J = computeCost(X, y, thetaNQ);
%fprintf('J with theta NQ = %f\n', J);
J = computeCost(X, y, thetaGD);
fprintf('J with theta GD = %f\n', J);

%figure;
%plot(X(:,2),y, 'rx', 'MarkerSize', 10);
%ylabel('Price');
%xlabel('Diamon weight in ca');




%theta = [0;0;0;0;0];
%J = computeCost(X, y, theta);

%theta = normalEqn(X, y);
%theta
                %case "FL": return 1;
                %case "IF": return 2;
                %case "VVS1": return 3;
                %case "VVS2": return 4;
                %case "VS1": return 5;
                %case "VS2": return 6;
                %case "SI1": return 7;
                %case "SI2": return 8;
                %case "I1": return 9;
                
                %case "D": return 1;
                %case "E": return 2;
                %case "F": return 3;
                %case "G": return 4;
                %case "H": return 5;
                %case "I": return 6;
                %case "J": return 7;

                %case "Astor Ideal": return 1;
                %case "Ideal": return 2;
                %case "Very Good": return 3;
                %case "Good": return 4;
                %case "Poor": return 5;



fprintf('Diamond 1\n');
%Clarity, Color, Cut
diamond = ([1 0.3^2 3^2 5 2] - mu)./sigma;

%fprintf('NQ res = %f\n', diamond' * thetaNQ);
fprintf('GD res = %f\n', diamond * thetaGD);
fprintf('Expected price $554\n');

fprintf('Diamond 2\n');
diamond = ([1 2.01^2 6^2 4 3] - mu)./sigma;
%fprintf('NQ res = %f\n', diamond' * thetaNQ);
fprintf('GD res = %f\n', diamond * thetaGD);
fprintf('Expected price $21,150\n');


fprintf('Diamond 3\n');
diamond = ([1 7.03^2 4^2 4 2] - mu)./sigma;
%fprintf('NQ res = %f\n', diamond' * thetaNQ);
fprintf('GD res = %f\n', diamond * thetaGD);
fprintf('Expected price $208,189\n');

figure;
[xx, yy] = meshgrid (X(:, 2), X(:, 3));
f = xx.^2 - yy.^2 + 2*xx.*yy.^2 + 1
surf(xx, yy, f);

