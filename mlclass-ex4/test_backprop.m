load('ex4data1.mat');

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;  

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

m = size(X, 1);

%Forward propagation
a1 = [ones(m,1) X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

for i = 1:size(Theta2,1) 
	Y(:,i) = y == i;
end;

%Backward propagationtion
d3 = a3 - Y;
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);
D2 = d3'*a2;
D1 = d2'*a1;
Theta1_grad = D2/m;
Theta2_grad = D1/m;