function [theta_hat,y_hat,Phat] = KFestimateSimpleGSCTAR(y,Hyperparam)

% Extracting the hyperparameter values
bar_theta = Hyperparam.bar_theta;           % Mean TAR parameter trajectory
vo = Hyperparam.vo;                         % Bias term
mu = Hyperparam.mu;                         % Stochastic constraint parameters
sigma_v = Hyperparam.sigma_v;               % Parameter innovations variance
sigma_w = Hyperparam.sigma_w;               % Innovations variance
N = length(y);                              % Signal length

% Setting up the Kalman filter matrices
xtt = zeros(1,N);                           % Posterior state vector estimate
xttm = zeros(1,N);                          % Prior state vector estimate
Ptt = zeros(1,N);                           % Posterior state error covariance matrix
Pttm = zeros(1,N);                          % Prior state error covariance matrix
K = zeros(1,N);                             % Kalman gain
yfiltered = zeros(1,N);                     % Filtered signal estimates

% Initial values
xtt(1:2) = bar_theta;                         % Initial state vector estimate
Ptt(1:2) = sigma_v;                           % Initial state error covariance matrix estimate
xttm(1:2) = bar_theta;                         % Initial state vector estimate
Pttm(1:2) = sigma_v;                           % Initial state error covariance matrix estimate

% Kalman filter
F = -mu;
R = sigma_w;
Q = sigma_v;
for i=2:N-1
    % System matrices
    H = -y(i-1);
    
    % Filter equations
    K(i) = Pttm(i)*H' / ( H*Pttm(i)*H' + R );
    xtt(i) = xttm(i) + K(i)*( y(i) - H*xttm(i) );
    Ptt(i) = Pttm(i) - K(i)*H'*Pttm(i);
    yfiltered(i) = H*xtt(i);
    
    % Update
    xttm(i+1) = F*xtt(i) + vo;
    Pttm(i+1) = F*Ptt(i)*F' + Q;
    
%     % Kalman gain
%     K(i) = Pttm(i)*H' / ( H*Pttm(i)*H' + R );
%     
%     % Correction
%     xtt(i) = xttm(i) + K(i) * ( y(i) - H*xttm(i)  );
%     yfiltered(i) = H*xtt(i);
%     Ptt(i) = ( 1 - K(i)*H )*Pttm(i);
end

theta_hat = xtt;
y_hat = yfiltered;
Phat = Ptt;