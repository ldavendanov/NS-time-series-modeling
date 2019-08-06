function [theta,e,criteria] = wls(Phi,Y,w)
%--------------------------------------------------------------------------
% Function to compute the Weighted Least Squares estimate of the parameter 
% vector theta for the problem
%                    Phi' * theta = Y
% where Phi is the M x N regression matrix, Y is the N x 1 response vector,
% and theta is the M x 1 parameter vector.
% Input :   Phi     ->  M x N regression matrix
%           Y       ->  N x 1 response vector
%           w       ->  N x 1 weighting vector (ones(N,1) for OLS estimate)
%
% Output :  theta   ->  WLS estimate of the parameter vector (M x 1 vector)
%           e       ->  Estimation residuals (N x 1 vector)
%           criteria->  Algorithm performance criteria
%           criteria.sigma_e  : Variance of the estimation residuals
%           criteria.Sigma_th : Parameter covariance matrix
%           criteria.CN       : Condition number of the regression matrix
%
% Created by :  David Avendano - April 2013 - Ver 1.0
%               All rights reserved
%--------------------------------------------------------------------------

% Gathering information from the input
N = size(Phi,2);
if nargin == 3  % Building the weighting matrix for the WLS estimate
    W = diag(1./w);
else    % If no third argument is provided, compute the OLS estimate
    W = eye(N);
end

% Computing the WLS estimate of theta
R = Phi*W*Phi';
theta = pinv( R )*( Phi*W*Y );
e = (Y - Phi'*theta)';

% Computing the performance criteria
criteria.sigma_e = var(e);
criteria.Sigma_th = pinv( R );
criteria.CN = cond( R );