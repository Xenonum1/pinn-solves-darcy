% Clear the workspace, command window, and close all figures
clear; clc; close all;
% Start the timer
tic

%% 1. Define Problem Parameters
% The Darcy's equation describes steady - state fluid flow in porous media.
% Combined with the mass conservation equation, it can be expressed as:
% ∇ · (k∇p) = 0   in Ω, where k is the permeability and p is the pressure.
% Assuming the permeability k = 1, the equation simplifies to the Laplacian equation ∇²p = 0   in Ω
% Boundary conditions are set as follows:
% - Left boundary x = 0: p = 1
% - Right boundary x = 1: p = 0
% - Top and bottom boundaries y = 0, y = 1: ∂p/∂n = 0

N_f = 1000; % Number of sampling points for the PDE, used to calculate the PDE residual
N_b = 100;  % Number of boundary points, used to satisfy the boundary conditions

%% 2. Build the Neural Network Model
layers = [
    featureInputLayer(2) % Input layer, receiving 2 - dimensional input corresponding to x and y coordinates in space
    fullyConnectedLayer(64) % Fully connected layer with 64 neurons
    tanhLayer() % Hyperbolic tangent activation function layer to add non - linearity to the network
    fullyConnectedLayer(64) % Fully connected layer with 64 neurons
    tanhLayer() % Hyperbolic tangent activation function layer
    fullyConnectedLayer(64) % Fully connected layer with 64 neurons
    tanhLayer() % Hyperbolic tangent activation function layer
    fullyConnectedLayer(1) % Output layer, outputting a 1 - dimensional result corresponding to the pressure p(x, y)
];
% Create a deep learning network based on the defined network layer structure
net = dlnetwork(layers); 

%% 3. Generate Training Data
% Sampling: Interior points (x, y)
x_f = rand(N_f, 1); % Randomly generate N_f x - coordinates in the interval [0, 1]
y_f = rand(N_f, 1); % Randomly generate N_f y - coordinates in the interval [0, 1]

% Boundary points (x, y)
% Sample boundary points from the four boundaries (left, right, top, bottom) respectively
x_left = zeros(N_b, 1);
y_left = linspace(0, 1, N_b)';
x_right = ones(N_b, 1);
y_right = linspace(0, 1, N_b)';
x_top_bottom = [linspace(0, 1, N_b)'; linspace(0, 1, N_b)'];
y_top_bottom = [ones(N_b, 1); zeros(N_b, 1)];

% Convert to dlarray format for automatic differentiation operations
x_f_dl = dlarray([x_f';y_f'], 'CB'); % Convert the coordinates of interior points to dlarray
x_left_dl = dlarray([x_left';y_left'], 'CB');
x_right_dl = dlarray([x_right';y_right'], 'CB');
x_top_bottom_dl = dlarray([x_top_bottom';y_top_bottom'], 'CB');

% Learning rate and training settings
numEpochs = 2000; % Total number of training epochs
learningRate = 0.001; % Learning rate, controlling the step size of parameter updates
% Pre - allocate an array to record the loss value of each training epoch
lossHistory = zeros(numEpochs,1); 

% Adam optimizer
averageGrad = []; % Initialize the first - moment estimate of the gradient
averageSqGrad = []; % Initialize the second - moment estimate of the gradient
beta1 = 0.9; % Exponential decay rate for the first - moment estimate in the Adam optimizer
beta2 = 0.999; % Exponential decay rate for the second - moment estimate in the Adam optimizer
epsilon = 1e-8; % Small constant to prevent division by zero

% Training loop
for epoch = 1:numEpochs
    % Call the modelLoss function for forward propagation to calculate the loss and gradients
    [loss, gradients] = dlfeval(@modelLoss, net, x_f_dl, x_left_dl, x_right_dl, x_top_bottom_dl);
    
    % Update the learnable parameters of the network using the Adam optimizer
    [net.Learnables, averageGrad, averageSqGrad] = adamupdate( ...
        net.Learnables, gradients, averageGrad, averageSqGrad, ...
        epoch, learningRate, beta1, beta2, epsilon);
    
    % Extract and record the loss value of the current epoch
    lossHistory(epoch) = extractdata(loss); 
    
    % Print the training progress every 100 epochs
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Loss = %.4e\n', epoch, lossHistory(epoch));
    end
end

% Visualize the predicted solution and the analytical solution
% Construct grid data
[X, Y] = meshgrid(linspace(0,1,100), linspace(0,1,100)); 
% Convert to dlarray format
input = dlarray([X(:)'; Y(:)'], 'CB'); 
% Predict the pressure
P_pred = extractdata(predict(net, input)); 
P_pred = reshape(P_pred, size(X));

% Analytical solution: p(x, y) = 1 - x
P_exact = 1 - X;

% Window 1: Predicted solution vs Analytical solution (side - by - side display)
% Set the size of the figure window
figure('Position', [100, 100, 1200, 600]); 

% Subplot 1: Numerical solution
subplot(1, 2, 1);
s1 = surf(X, Y, P_pred, 'EdgeColor', 'none', 'FaceLighting', 'gouraud');
title('PINN Numerical Solution p(x,y)', 'FontSize', 14);
xlabel('x', 'FontSize', 12); ylabel('y', 'FontSize', 12); zlabel('p(x,y)', 'FontSize', 12);
view(3); colorbar; colormap jet;
camlight left; lighting gouraud;
grid on;

% Subplot 2: Analytical solution
subplot(1, 2, 2);
s2 = surf(X, Y, P_exact, 'EdgeColor', 'none', 'FaceLighting', 'gouraud');
title('Analytical Solution p(x,y) = 1 - x', 'FontSize', 14);
xlabel('x', 'FontSize', 12); ylabel('y', 'FontSize', 12); zlabel('p(x,y)', 'FontSize', 12);
view(3); colorbar; colormap jet;
camlight right; lighting gouraud;
grid on;

% Synchronize the viewing angles
linkaxes([subplot(1,2,1), subplot(1,2,2)]); 
sgtitle('Comparison of Darcy Equation Solving Results');

% Window 2: Relative error distribution
figure('Position', [100, 100, 800, 600]);

% Calculate the relative error, add a very small value to prevent division by zero
relative_error = abs(P_pred - P_exact);

% Use imagesc to display the 2D error map (surf is optional)
imagesc(relative_error, 'XData', [0 1], 'YData', [0 1]);
axis xy; % Keep the image consistent with the coordinate direction
colorbar;
colormap hot;
title('Absolute Error Distribution |p_{PINN} - p_{Exact}|', 'FontSize', 14);
xlabel('x', 'FontSize', 12); ylabel('y', 'FontSize', 12);
grid on;

% Use surf:
% surf(X, Y, relative_error, 'EdgeColor','none');
% title('Relative Error Distribution');
% xlabel('x'); ylabel('y'); zlabel('Relative Error');
% colorbar; colormap hot;
% Stop the timer and display the elapsed time
toc

%% Sub - function: Loss calculation
function [loss, gradients] = modelLoss(net, x_f, x_left, x_right, x_top_bottom)
    % PDE residual
    % Forward propagation to calculate the network output at interior points
    u = forward(net, x_f); 
    
    % Calculate the first - order partial derivatives of u with respect to x and y
    du_x = dlgradient(sum(u, 'all'), x_f(:,1,:), 'EnableHigherDerivatives', true);
    du_y = dlgradient(sum(u, 'all'), x_f(:,2,:), 'EnableHigherDerivatives', true);
    
    % Calculate the second - order partial derivatives of u with respect to x and y
    du_xx = dlgradient(sum(du_x, 'all'), x_f(:,1,:), 'EnableHigherDerivatives', true);
    du_yy = dlgradient(sum(du_y, 'all'), x_f(:,2,:), 'EnableHigherDerivatives', true);
    
    % Calculate the PDE residual
    residual = -du_xx - du_yy; 
    % Calculate the PDE residual loss
    loss_PDE = mean(residual.^2); 

    % Left and right boundary loss
    u_left = forward(net, x_left);
    u_right = forward(net, x_right);
    % Calculate the left and right boundary condition loss
    loss_BC_left_right = mean((u_left - 1).^2) + mean(u_right.^2); 
    
    % Top and bottom boundary loss (Neumann condition)
    u_top_bottom = forward(net, x_top_bottom);
    du_top_bottom_x = dlgradient(sum(u_top_bottom, 'all'), x_top_bottom(:,1,:), 'EnableHigherDerivatives', true);
    % Calculate the top and bottom boundary condition loss
    loss_BC_top_bottom = mean(du_top_bottom_x.^2); 
    
    % Calculate the total loss
    loss = loss_PDE + loss_BC_left_right + loss_BC_top_bottom; 
    % Calculate the gradients of the total loss with respect to the learnable parameters of the network
    gradients = dlgradient(loss, net.Learnables); 
end
