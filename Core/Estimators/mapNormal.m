function M = mapNormal(Phi,Y,Theta0,SigmaTh,sigmaW2)

[n,N] = size(Phi);

%-- Recursive calcualtion of the posterior mean and parameter covariance
P_ = SigmaTh;
theta_ = Theta0;
I = eye(n);

err = zeros(1,N);
sigmaE2 = zeros(1,N);
theta = zeros(n,N);
P = zeros(n,n,N);
traceP = zeros(1,N);
normK = zeros(1,N);

%-- Running the recursive MAP estimator
for t=1:N
    
    %-- A priori estimation error and a priori error covariance matrix
    err(t) = Y(t) - Phi(:,t)'*theta_;                                       % A priori prediction error
    sigmaE2(t) = sigmaW2 + Phi(:,t)'*P_*Phi(:,t);                            % Prior error variance
    
    %-- Gain
    K = P_*Phi(:,t)/sigmaE2(t);
    
    %-- Update MAP parameter vector estimate and its covariance matrix
    theta(:,t) = theta_ + K*err(t);
    P(:,:,t) = ( I - K*Phi(:,t)' )*P_;
    
    %-- Diagnostics
    traceP(t) = trace(P(:,:,t));
    normK(t) = norm(K);
    
    %-- Updating prior estimates
    theta_ = theta(:,t); 
    P_ = P(:,:,t);
    
end

%-- Computing the OLS parameter estimates
M.ParameterVector = theta(:,N)';                                            % Parameter vector estimate
Yhat = M.ParameterVector*Phi;                                               % One-step-ahead predictions
err = Y - Yhat;                                                             % One-step-ahead prediction error
sigmaW2 = var(err);
M.InnovationsVariance = sigmaW2;                                            % Innovations variance

%-- Calculating the parameter covariance matrix estimate
M.ParameterCovariance = P(:,:,N);                                           % Estimated covariance matrix of the parameter vector

%-- Performance criteria
M.performance.rss = sum(err.^2);                                            % Residual Sum of Squares ( RSS )
M.performance.rss_sss = M.performance.rss/sum(Y.^2);                        % Residual Sum of Squares over Series Sum of Squares
M.performance.lnL = -(1/2)*( sum( log(2*pi*sigmaW2) + err.^2/sigmaW2 ) );   % Log-likelihood
M.performance.bic = log(N)*n - 2*M.performance.lnL;                         % Bayesian Information Criterion ( BIC )
M.performance.spp = N/n;                                                    % Samples Per Parameter ( SPP )

%-- Diagnostic of the estimator
sigmaTheta2 = diag(M.ParameterCovariance);                                  % Diagonal of the perameter covariance
M.performance.chi2_theta = M.ParameterVector(:).^2 ./ sigmaTheta2;          % Statistic to determine if the parameters are statistically different from zer