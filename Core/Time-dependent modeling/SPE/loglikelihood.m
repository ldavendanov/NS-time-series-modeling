function L = loglikelihood(e,se)
%--------------------------------------------------------------------------
% Function to compute the log likelihood of a model in terms of the
% innovations (residuals) 'e' and innovations variance 'se'
%
% Created by :  David Avendano - April 2013 - Ver 1.0
%               All rights reserved
%--------------------------------------------------------------------------

N = length(e);
L = -N*log(2*pi)/2 - sum( log(se) + e.^2./se )/2;