function [theta_hat,P_theta,y_hat,HyperPar,other] = GSCTAR_EM2(y,na,q)
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

% HyperPar0 = [mu0; theta0; log(sw0); log(sv0)];      % Initial hyperparameter vector

% Finding initial values with conventional EM
optim = 'full';
[~,~,~,HyperPar] = GSCTAR_EM(y,na,q);
if strcmp(optim,'full')
    HyperPar0 = HyperPar(1:na+q+1,end);
    HyperPar0(q+na+1) = log(HyperPar0(q+na+1));
    switch q
        case 1 
            P.sv = 1e-4;
        case 2
            P.sv = 1e-6;
        case 3
            P.sv = 1e-8;
    end
else
    HyperPar0 = HyperPar(1:q,end);
    P.theta0 = HyperPar(q+(1:na),end);
    P.sw = HyperPar(q+na+1,end);
    switch q
        case 1 
            P.sv = 1e-4;
        case 2
            P.sv = 1e-6;
        case 3
            P.sv = 1e-8;
    end
end


problem.objective = @(x)ExpectationStep(y,na,q,x,P);
problem.X0 = HyperPar0;
problem.solver = 'fminsearch';
optim_opts = optimset('fminsearch');
optim_opts.Display = 'iter';
optim_opts.MaxIter = 2e2;
optim_opts.MaxFunEvals = 1e3;
optim_opts.TolFun = 1e-2;
optim_opts.TolX = 1e-2;
problem.options = optim_opts;
HyperPar = fminsearch(problem);

% Computing for the optimized parameter values
options.estim = 'ks';
if strcmp(optim,'full')
    HP.mu = HyperPar(1:q)';                             % Optimized stochastic constraint vector
    HP.theta0 = HyperPar(q+(1:na));                     % Optimized parameter mean
    HP.sigma_w2 = exp(HyperPar(q+na+1));                % Optimized innovations variance
    HP.Sigma_v = P.sv*eye(na);
else
    HP.mu = HyperPar(1:q)';                             % Optimized stochastic constraint vector
    HP.theta0 = P.theta0;                               % Optimized parameter mean
    HP.sigma_w2 = P.sw;                                 % Optimized innovations variance
    HP.Sigma_v = P.sv*eye(na);
end
[theta_hat,P_theta,y_hat,other] = GSCTAR_MAPtrajectory(y,na,HP,options);

%--------------------------------------------------------------------------
function [Q,theta_hat,P_theta,y_hat,other] = ExpectationStep(y,na,q,HP,P)

sv = P.sv;

% Obtaining information from the input
if numel(HP)==na+q+1
    Hyperpar.mu = HP(1:q)';
    Hyperpar.theta0 = HP(q+(1:na));
    Hyperpar.sigma_w2 = exp(HP(q+na+1));
    Hyperpar.Sigma_v = sv*eye(na);
elseif numel(HP)==q,
    Hyperpar.mu = HP(1:q)';
    Hyperpar.theta0 = P.theta0;
    Hyperpar.sigma_w2 = P.sw;
    Hyperpar.Sigma_v = sv*eye(na);
end
options.estim = 'kc';

% Making some "security" checks
rho = roots([1 Hyperpar.mu]);
cond1 = min(abs(rho<=1));           % Condition 1 : The roots of the stochastic constraints have magnitude lower than or equal to one
rho = roots([1 Hyperpar.theta0']);
cond2 = min(abs(rho<1));            % Condition 2 : The roots of the mean (initial) parameter vector have magnitude lower than one

% Evaluating the log-likelihood
if cond1 && cond2
    % Computing estimates
    [theta_hat,P_theta,y_hat,other] = GSCTAR_MAPtrajectory(y,na,Hyperpar,options);
    Q = other.Q;
else
    Q = inf;
end
