function [theta_hat,P_theta,y_hat,HyperPar,other] = SPTARMA_MLII(y,na,nc,q,options)
%--------------------------------------------------------------------------
% This function computes ML-II estimates of the hyperparameters of an
% SP-TAR model, including the innovations variance and the parameter
% innovations covariance matrix. It also provides the MAP parameter
% trajectories using the KF and KS based on the optimized hyperparameters.
% Input parameters:
%   y       -> Observation vector ( vector : 1 x N )
%   na      -> AR order ( scalar )
%   nc      -> MA order ( scalar )
%   q       -> Smoothness priors order ( scalar )
%   options -> Estimator options
%
% Output parameters:
%   theta_hat -> Estimated parameter trajectories ( matrix : na+nc x N )
%   P_theta   -> (Smoothed) parameter estimation error covariance ( 3D matrix : na+nc x na+nc x N )
%   y_hat     -> One-step-ahead signal predictions ( vector : 1 x N )
%   HyperPar  -> Structure with the hyperparameters
%
%--------------------------------------------------------------------------

% Setting up the initial values for the optimization
data = iddata(y',[],1);
sys = armax(data,[na nc]);
theta0 = [sys.a(2:end)'; sys.c(2:end)'];
sw0 = sys.NoiseVariance;
sv0 = 1e-6;                                         % Initial parameter innovations covariance
HyperPar0 = [theta0; log10(sw0); log10(sv0)];       % Initial hyperparameter vector

% Setting up the non-linear optimization
problem.objective = @(x)SPTARMAlogMarginal(y,na,nc,q,x);
problem.X0 = HyperPar0;                             % Initial guess of the hyperparameter vector
problem.solver = 'patternsearch';
optim_opts = psoptimset;
optim_opts.Display = 'iter';
optim_opts.CompletePoll = 'on';
optim_opts.InitialMeshSize = 1;
optim_opts.MaxIter = 50;                            % Maximum number of iterations of the optimization
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
    problem.options = optim_opts;
    HyperPar = fminsearch(problem);
end

% Computing for the optimized parameter values
options.estim = 'ks';
mu0 = poly(ones(1,q));
mu0 = mu0(2:end);                                   % Stochastic constraint parameter vector
HP.mu = mu0;                                        % Stochastic constraint parameters
HP.theta0 = HyperPar(1:na+nc);                      % Optimized parameter mean
HP.sigma_w2 = 10^HyperPar(end-1);                   % Optimized innovations variance
HP.Sigma_v = (10^HyperPar(end))*eye(na+nc);         % Optimized parameter innovations variance
[theta_hat,P_theta,y_hat,other] = GSCTARMA_MAPtrajectory(y,na,nc,HP,options);

%--------------------------------------------------------------------------
function [logL,theta_hat,P_theta,y_hat,other] = SPTARMAlogMarginal(y,na,nc,q,HP)

% Obtaining information from the input
mu0 = poly(ones(1,q));
mu0 = mu0(2:end);
Hyperpar.mu = mu0;
Hyperpar.theta0 = HP(1:na+nc);
Hyperpar.sigma_w2 = 10^HP(end-1);
Hyperpar.Sigma_v = (10^HP(end))*eye(na+nc);

% Making some "security" checks
rho = roots([1 Hyperpar.theta0']);
cond = min(abs(rho<1));            % Condition 2 : The roots of the mean (initial) parameter vector have magnitude lower than one

% Evaluating the log-likelihood
if cond
    [theta_hat,P_theta,y_hat,other] = GSCTARMA_MAPtrajectory(y,na,nc,Hyperpar);
    logL = other.logL;
else
    logL = inf;
end