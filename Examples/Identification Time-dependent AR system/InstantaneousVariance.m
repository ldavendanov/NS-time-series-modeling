function M = InstantaneousVariance( y, M, Phi, Gbs )

err = y - M.Parameters.Theta*Phi;

S = ols(Gbs,err.^2);
M.InnovationsVariance.S = S;