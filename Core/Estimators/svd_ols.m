function M = svd_ols(Phi,Y)

m = size(Y,1);
[n,N] = size(Phi);
[U,S,V] = svd(Phi,'econ');
L = diag(1./diag(S));

%-- Computing the OLS parameter estimates
Theta = Y*V*L*U';                                                           % Parameter vector estimate
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
K0 = U*L.^2*U';
if m == 1
    SigmaTheta = sigmaW2*eye(n)*K0;                                         % Estimated covariance matrix of the parameter vector
    sigmaTheta2 = diag(SigmaTheta);                                         % Diagonal of the perameter covariance
else
    SigmaTheta.K0 = K0;                                                     % Estimated covariance matrix of the parameter vector
    SigmaTheta.SigmaW = SigmaW;
    sigmaTheta2 = kron(diag(K0),diag(SigmaW));                              % Diagonal of the perameter covariance
end

%-- Performance criteria
M.Performance.rss = sum(sum(err.^2));                                       % Residual Sum of Squares ( RSS )
M.Performance.rss_sss = M.Performance.rss/sum(sum(Y.^2));                   % Residual Sum of Squares over Series Sum of Squares

if m==1                                                                     % Log-likelihood
    M.Performance.lnL = -(1/2)*( sum( log(2*pi*sigmaW2) + err.^2/sigmaW2 ) );
else
    M.Performance.lnL = -(1/2)*( trace( log(2*pi*det(SigmaW)) + (err*err')/SigmaW ) );
end

%-- Diagnostic of the estimator
M.Performance.bic = log(N*n)*numel(Theta) - 2*M.Performance.lnL;            % Bayesian Information Criterion ( BIC )
M.Performance.spp = N*n/numel(Theta);                                       % Samples Per Parameter ( SPP )
M.Performance.CN = cond(K0);                                                % Condition Number of the inverted matrices - Numerical accurady of the estimate
M.Performance.chi2_theta = Theta(:).^2 ./ sigmaTheta2;                      % Statistic to determine if the parameters are statistically different from zer

%-- Packing the output
M.Parameters.Theta = Theta;
M.Parameters.SigmaTheta = SigmaTheta;