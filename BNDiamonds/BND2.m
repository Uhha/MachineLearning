
%clear;
%data = load('BNDiamonds.txt');


y = data(:, 1);
m = length(y); % number of training examples

%X =  [ones(m, 1), data(:, 2:5)]; 
X =  [data(:, 2:5)]; 



input_layer_size  = 4;  
hidden_layer_size = 2;   
outer_layer_size = 1;          



initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, outer_layer_size);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

checkNNGradients;
lambda = 3;
checkNNGradients(lambda);

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 50);
lambda = 1;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   outer_layer_size, X, y, lambda);
                                   
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 outer_layer_size, (hidden_layer_size + 1));
                 
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);