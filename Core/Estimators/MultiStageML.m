function M = MultiStageML( Phi, Y, Gbs )
%--------------------------------------------------------------------------
% Multi-Stage Weighted Least Squares method for estimation of the
% projection coefficients of the parameters and innovations variance of
% Functional Series Time-dependent AR and Linear-Parameter-Varying AR
% models.
%--------------------------------------------------------------------------


%-- Part 1 : Initial values -----------------------------------------------
% Initial values of the coefficients of projection of the AR parameters and
% innovations variance are obtained. OLS is used to calculate coefficients
% of projection of the AR parameters, IV (instantaneous variance) method is
% used to calculate the coefficients of projection of the innovations
% variance.

% OLS for AR parameter estimates
m0 = ols( Phi, Y ); 

% IV for innovations variance estimates
m0 = InstantaneousVariance( Y, m0, Phi, Gbs );
m0.InnovationsVariance.s = m0.InnovationsVariance.S.Parameters.Theta;
m0.InnovationsVariance.sigmaW2 = m0.InnovationsVariance.s*Gbs;
theta0 = m0.Parameters.Theta;

%-- Part 2 : Loop of the Multi-Stage WLS method ---------------------------
% Initial coefficients are iteratively refined by means of the Multi-Stage
% WLS optimization method.

fprintf('Multi-Stage WLS optimization\n')
fprintf('Iteration \t RSS/SSS \t\t  | D_Theta | \n')
for i=1:10
    
    %-- WLS for estimation of AR coefficients of projection
    W = diag(m0.InnovationsVariance.sigmaW2);                               % WLS weighting matrix
    M = wls( Phi, Y, W );                                                   % Calcualte WLS
    
    %-- IV for innovations variance coefficients of projection
    M = InstantaneousVariance( Y, M, Phi, Gbs );                            % IV estimation
    
    %-- Extract updated coefficient vector
    theta = M.Parameters.Theta;
    
    fprintf('%3d \t\t %2.4e \t %2.4e \n',i,M.Performance.rss_sss,norm(theta-theta0))
    
    m0 = M;                                                                 % Update previous model
    theta0 = m0.Parameters.Theta;                                           % Update previous coefficient vector estimate
end