function M = ols(Phi,Y)

[n,N] = size(Phi);

%-- Computing the OLS parameter estimates
M.ParameterVector = Y/Phi;                                                  % Parameter vector estimate
Yhat = M.ParameterVector*Phi;                                               % One-step-ahead predictions
err = Y - Yhat;                                                             % One-step-ahead prediction error
sigmaW2 = var(err);
M.InnovationsVariance = sigmaW2;                                            % Innovations variance

%-- Calculating the parameter covariance matrix estimate
M.FisherInformation = Phi*Phi';                                             % Fisher Information matrix
M.ParameterCovariance = sigmaW2*eye(n)/(Phi*Phi');                          % Estimated covariance matrix of the parameter vector

%-- Performance criteria
M.performance.rss = sum(err.^2);                                            % Residual Sum of Squares ( RSS )
M.performance.rss_sss = M.performance.rss/sum(Y.^2);                        % Residual Sum of Squares over Series Sum of Squares
M.performance.lnL = -(1/2)*( sum( log(2*pi*sigmaW2) + err.^2/sigmaW2 ) );   % Log-likelihood
M.performance.bic = log(N)*n - 2*M.performance.lnL;                         % Bayesian Information Criterion ( BIC )
M.performance.spp = N/n;                                                    % Samples Per Parameter ( SPP )
M.performance.CN = cond(M.FisherInformation);                               % Condition Number of the inverted matrices - Numerical accurady of the estimate

%-- Diagnostic of the estimator
sigmaTheta2 = diag(M.ParameterCovariance);                                  % Diagonal of the perameter covariance
M.performance.chi2_theta = M.ParameterVector(:).^2 ./ sigmaTheta2;          % Statistic to determine if the parameters are statistically different from zer