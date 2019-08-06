function M = wls(Phi,Y,W)

m = size(Y,1);
[n,N] = size(Phi);

%-- Computing the WLS parameter estimates
Theta = Y*W*Phi'/(Phi*W*Phi');                                              % Parameter vector estimate
Yhat = Theta*Phi;                                                           % One-step-ahead predictions
err = Y - Yhat;                                                             % One-step-ahead prediction error

if m == 1
    sigmaW2 = var(err);
    M.InnovationsVariance.sigmaW2 = sigmaW2;                                % Innovations variance (scalar output)
else
    SigmaW = cov(err');
    M.InnovationsCovariance.SigmaW = SigmaW;                                % Innovations covariance matrix (vector output)
end

%-- Calculating the parameter covariance matrix estimate
C0 = Phi*Phi';
if m == 1
    SigmaTheta = sigmaW2*eye(n)/C0;                                         % Estimated covariance matrix of the parameter vector
else
    K0 = pinv(C0);
    SigmaTheta = kron(K0,SigmaW);                                           % Estimated covariance matrix of the parameter vector
end

%-- Performance criteria
M.Performance.rss = sum(sum(err.^2));                                       % Residual Sum of Squares ( RSS )
M.Performance.rss_sss = M.Performance.rss/sum(sum(Y.^2));                   % Residual Sum of Squares over Series Sum of Squares

if m==1                                                                     % Log-likelihood
    M.Performance.lnL = -(1/2)*( sum( log(2*pi*sigmaW2) + err.^2/sigmaW2 ) );
else
    M.Performance.lnL = -(1/2)*( trace( log(2*pi*det(SigmaW)) + (err'/SigmaW)*err ) );
end

%-- Diagnostic of the estimator
M.Performance.bic = log(N)*n - 2*M.Performance.lnL;                         % Bayesian Information Criterion ( BIC )
M.Performance.spp = N/n;                                                    % Samples Per Parameter ( SPP )
M.Performance.CN = cond(C0);                                                % Condition Number of the inverted matrices - Numerical accurady of the estimate
sigmaTheta2 = diag(SigmaTheta);                                             % Diagonal of the perameter covariance
M.Performance.chi2_theta = Theta(:).^2 ./ sigmaTheta2;                      % Statistic to determine if the parameters are statistically different from zer

%-- Packing the output
M.Parameters.Theta = Theta;
M.Parameters.SigmaTheta = SigmaTheta;