function M = MultiStageML( Phi, Y, Gbs )

%-- Initial OLS estimate of the AR parameter vector
m0 = ols( Phi, Y );

%-- Initial Instantaneous Variance estimate
m0 = InstantaneousVariance( Y, m0, Phi, Gbs );
m0.InnovationsVariance.s = m0.InnovationsVariance.S.Parameters.Theta;
m0.InnovationsVariance.sigmaW2 = m0.InnovationsVariance.s*Gbs;
theta0 = m0.Parameters.Theta;

%-- Re-calculating the AR parameter vector
figure
for i=1:5
    W = diag(m0.InnovationsVariance.sigmaW2);
    M = wls( Phi, Y, W );
    M = InstantaneousVariance( Y, M, Phi, Gbs );
    theta = M.Parameters.Theta;
    
    subplot(211)
    semilogy( i, M.Performance.rss_sss,'.b' )
    hold on
    
    subplot(212)
    semilogy( i, norm( theta - theta0 ),'.b' )
    hold on
    drawnow
    
    m0 = M;
    theta0 = m0.Parameters.Theta;
end