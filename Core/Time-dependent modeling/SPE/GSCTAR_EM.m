function [theta_hat,P_theta,y_hat,HyperPar,other] = GSCTAR_EM(y,na,q)
%--------------------------------------------------------------------------
% This function computes ML estimates of the hyperparameters of a GSC-TAR
% model, including the stochastic constraint parameters, the innovations
% variance and the parameter innovations covariance matrix via EM
% algorithm. It also provides the MAP parameter trajectories using the KF
% based on the optimized hyperparameters.
% Input parameters:
%   y       -> Observation vector ( vector : 1 x N )
%   na      -> Model order ( scalar )
%   q       -> Stochastic constraint order ( scalar )
%   options -> Estimator options
%
% Output parameters:
%   theta_hat -> Estimated parameter trajectories ( matrix : na x N )
%   P_theta   -> (Smoothed) parameter estimation error covariance ( 3D matrix : na x na x N )
%   y_hat     -> One-step-ahead signal predictions ( vector : 1 x N )
%   HyperPar  -> Structure with the hyperparameters
%
%--------------------------------------------------------------------------

% Setting up the computation matrices
MaxIter = 8;
HyperPar = zeros(q+na+2,MaxIter);
Q = zeros(1,MaxIter);

% Setting up the initial values for the optimization
mu0 = poly([0.98 0.9*ones(1,q-1)]);
mu0 = mu0(2:end)';                                  % Initial stochastic constraint parameter values
[theta0,sw0] = arburg(y,na);                        % Initial parameter mean and innovations variance values
theta0 = theta0(2:end)';
switch q
    case 1
        sv0 = 1e-4;                                         % Initial parameter innovations covariance
    case 2
        sv0 = 1e-6;                                         % Initial parameter innovations covariance
    case 3
        sv0 = 1e-8;                                         % Initial parameter innovations covariance
end
HyperPar(:,1) = [mu0; theta0; sw0; sv0];            % Initial hyperparameter vector

% Iterating through the EM algorithm
fprintf('Optimizing hyperaparameters via EM\n')
for j=1:MaxIter,
    [Q(j),theta_hat,P_theta,y_hat,other] = ExpectationStep(y,na,q,HyperPar(:,j));
    HyperPar(:,j+1) = MaximizationStep(theta_hat,na,q,HyperPar(:,j),other);
    fprintf('Iteration No. %4d, Expected log-likelihood %1.3e \n',j,Q(j))
end
fprintf('Done!!\n')

%--------------------------------------------------------------------------
function [Q,theta_hat,P_theta,y_hat,other] = ExpectationStep(y,na,q,HP)

% Obtaining information from the input
Hyperpar.mu = HP(1:q)';
Hyperpar.theta0 = HP(q+(1:na));
Hyperpar.sigma_w2 = HP(end-1);
Hyperpar.Sigma_v = HP(end)*eye(na);
options.estim = 'kc';

[theta_hat,P_theta,y_hat,other] = GSCTAR_MAPtrajectory(y,na,Hyperpar,options);
Q = other.Q;

%--------------------------------------------------------------------------
function HyperPar = MaximizationStep(theta_hat,na,q,HyperPar_,other)

% Obtaining information from the input
N = size(theta_hat.smooth,2);
mu_ = HyperPar_(1:q)';

% Updating the mean/initial parameter vector
rho = roots([1 mu_]);
cond1 = min(abs(rho<=1));           % Condition 1 : The roots of the stochastic constraints have magnitude lower than or equal to one
if cond1
    theta0 = mean(theta_hat.smooth,2);
else
    theta0 = theta_hat.smooth(:,1);
end

HyperPar = HyperPar_;
HyperPar(q+(1:na)) = theta0;
HyperPar(q+na+1) = other.C/(N-2*na);

%--------------------------------------------------------------------------
% % Updating the stochastic constraing parameters
% theta0_ = HyperPar_(q+(1:na));
% sigma_w2_ = HyperPar_(end-1);
% sigma_v_ = HyperPar_(end);
% S11 = other.S11;
% S00 = other.S00;
% S10 = other.S10;
% Svz = other.Svz;
%
% barS00 = zeros(q);
% barS10 = zeros(q);
% for k=1:q
%     indK = (1:na)+na*(k-1);
%     for j=1:q,
%         indJ = (1:na)+na*(j-1);
%         barS00(k,j) = trace(S00(indK,indJ));
%         barS10(k,j) = trace(0.5*S10(indK,indJ)+0.5*S10(indJ,indK)+Svz(indK,indJ));
%     end
% end
% mu = -barS10(1,:)/barS00;

% sigma_v2 = trace(S11-(S10/S00)*S10')/(na*q*(N-2*na));

% HyperPar = HyperPar_;
% HyperPar(1:q) = mu';
% HyperPar(q+(1:na)) = theta0;
% HyperPar(q+na+1) = other.C/(N-2*na);
% if sigma_v2 > 1e-10
%     HyperPar(q+na+2) = sigma_v2;
% end