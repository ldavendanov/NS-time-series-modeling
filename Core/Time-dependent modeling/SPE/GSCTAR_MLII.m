function [theta_hat,P_theta,y_hat,HyperPar,other] = GSCTAR_MLII(y,na,q,options)
%--------------------------------------------------------------------------
% This function computes ML-II estimates of the hyperparameters of a
% GSC-TAR model, including the stochastic constraint parameters, the
% innovations variance and the parameter innovations covariance matrix. It
% also provides the MAP parameter trajectories using the KF based on the
% optimized hyperparameters.
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

% Setting up the initial values for the optimization
mu0 = poly(0.9*ones(1,q-1));
mu0 = mu0(2:end)';                                  % Initial stochastic constraint parameter values
[theta0,sw0] = arburg(y,na);                        % Initial parameter mean and innovations variance values
theta0 = theta0(2:end)';
switch q
    case 1
        sv0 = 1e-2;                                 % Initial parameter innovations covariance
    case 2
        sv0 = 1e-4;                                 % Initial parameter innovations covariance
    case 3
        sv0 = 1e-6;                                 % Initial parameter innovations covariance
end
HyperPar0 = [mu0; log(sw0); log(sv0)];              % Initial hyperparameter vector

% Setting up the non-linear optimization
problem.objective = @(x)GSCTARlogMarginal(y,na,q,x,theta0);
problem.X0 = HyperPar0;                             % Initial guess of the hyperparameter vector
problem.solver = 'patternsearch';
optim_opts = psoptimset;
optim_opts.Display = 'iter';
optim_opts.CompletePoll = 'on';
optim_opts.InitialMeshSize = 1;
optim_opts.MaxIter = 30;                            % Maximum number of iterations of the optimization
optim_opts.UseParallel = 1;
optim_opts.Vectorized = 'off';
problem.options = optim_opts;

% Finding a feasible area using the patternsearch algorithm
[HyperPar,~,exitflag] = patternsearch(problem);

% Refining the estimate with fmincon if patternsearch does not reach convergence
if exitflag <= 0
    problem.X0 = HyperPar;
    problem.solver = 'fminsearch';
    optim_opts = optimset('fminsearch');
    optim_opts.Display = 'iter';
    optim_opts.MaxIter = 1e3;
    optim_opts.MaxFunEvals = 5e3;
    problem.options = optim_opts;
    HyperPar = fminsearch(problem);
end

% Computing for the optimized parameter values
options.estim = 'ks';
rho = roots([1; HyperPar(1:q-1)]);
mu = poly([1; rho]);
HP.mu = mu(2:end);                                  % Optimized stochastic constraint vector
HP.theta0 = theta0;                                 % Optimized parameter mean
HP.sigma_w2 = exp(HyperPar(end-1));                 % Optimized innovations variance
HP.Sigma_v = exp(HyperPar(end))*eye(na);            % Optimized parameter innovations variance
[theta_hat,P_theta,y_hat,other] = GSCTAR_MAPtrajectory(y,na,HP,options);

% Organizing the output
HyperPar = [HP.mu'; HP.theta0; log(HP.sigma_w2); log(HP.Sigma_v(1,1))];

%--------------------------------------------------------------------------
function [logL,theta_hat,P_theta,y_hat,other] = GSCTARlogMarginal(y,na,q,HP,theta0)

% Obtaining information from the input
rho = roots([1; HP(1:q-1)]);
mu = poly([1; rho]);
Hyperpar.mu = mu(2:end);
Hyperpar.theta0 = theta0;
Hyperpar.sigma_w2 = exp(HP(end-1));
Hyperpar.Sigma_v = exp(HP(end))*eye(na);

% Making some "security" checks
rho = roots([1 Hyperpar.mu]);
cond1 = min(abs(rho<=1));           % Condition 1 : The roots of the stochastic constraints have magnitude lower than or equal to one
cond4 = min(real(rho))>0.8;
rho = roots([1 Hyperpar.theta0']);
cond2 = min(abs(rho<1));            % Condition 2 : The roots of the mean (initial) parameter vector have magnitude lower than one
cond3 = HP(end) > log(5e-6);
% cond3 = 1;

% Evaluating the log-likelihood
if cond1 && cond2 && cond3 && cond4
    [theta_hat,P_theta,y_hat,other] = GSCTAR_MAPtrajectory(y,na,Hyperpar);
    logL = other.logL;
else
    logL = inf;
end