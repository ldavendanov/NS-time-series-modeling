function M = mapNormalBatch(Phi,Y,Theta0,Lambda0,SigmaW20,nu0)

m = size(Y,1);
[n,N] = size(Phi);

%-- Batch computation of posterior parameter mean and covariance
Lambda = Phi*Phi' + Lambda0;
Theta = (Y*Phi' + Theta0*Lambda0) / Lambda;

errY = Y - Theta*Phi;
errTh = Theta - Theta0;

V = SigmaW20 + errY*errY' + ( errTh/Lambda0 )*errTh';
nu = nu0 + N;
SigmaW = V/(nu + m + 1);
if m == 1
    M.InnovationsVariance.sigmaW2 = SigmaW;                                 % Innovations variance (scalar output)
else
    M.InnovationsCovariance.SigmaW = SigmaW;                                % Innovations covariance matrix (vector output)
end

%-- Calculating the parameter covariance matrix estimate
if m == 1
    SigmaTheta = sigmaW2*eye(n)/Lambda;                                     % Estimated covariance matrix of the parameter vector
else
    SigmaTheta.Lambda = Lambda;                                                 % Estimated covariance matrix of the parameter vector
    SigmaTheta.SigmaW = SigmaW;
    sigmaTheta2 = kron(diag(Lambda),diag(SigmaW));                          % Diagonal of the perameter covariance
end

%-- Performance criteria
M.Performance.rss = sum(sum(errY.^2));                                      % Residual Sum of Squares ( RSS )
M.Performance.rss_sss = M.Performance.rss/sum(sum(Y.^2));                   % Residual Sum of Squares over Series Sum of Squares

if m==1                                                                     % Log-likelihood
    M.Performance.lnL = -(1/2)*( sum( log(2*pi*sigmaW2) + errY.^2/sigmaW2 ) );
else
    M.Performance.lnL = -(1/2)*( trace( log(2*pi*det(SigmaW)) + (errY*errY')/SigmaW) );
end

%-- Diagnostic of the estimator
M.Performance.bic = log(N)*n - 2*M.Performance.lnL;                         % Bayesian Information Criterion ( BIC )
M.Performance.spp = N/n;                                                    % Samples Per Parameter ( SPP )
M.Performance.CN = cond(Lambda);                                            % Condition Number of the inverted matrices - Numerical accurady of the estimate
M.Performance.chi2_theta = Theta(:).^2 ./ sigmaTheta2;                      % Statistic to determine if the parameters are statistically different from zer

%-- Packing the output
M.Parameters.Theta = Theta;
M.Parameters.SigmaTheta = SigmaTheta;