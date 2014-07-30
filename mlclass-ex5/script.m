
load ('ex5data1.mat');
lambda = 1
m = size(X, 1);
theta = trainLinearReg([ones(m,1) X], y, lambda)