function [theta_hat,P_theta,y_hat,HyperPar,other] = GSCTARMA_MLII(y,na,nc,q,options)
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
mu0 = poly([1 0.9*ones(1,q-1)]);
mu0 = mu0(2:end)';                                  % Initial stochastic constraint parameter values
data = iddata(y',[],1);
sys = armax(data,[na nc]);
theta0 = [sys.a(2:end)'; sys.c(2:end)'];
sw0 = sys.NoiseVariance;
sv0 = 1e-6;                                         % Initial parameter innovations covariance
HyperPar0 = [mu0; theta0; log(sw0); log(sv0)];      % Initial hyperparameter vector

% Setting up the non-linear optimization
problem.objective = @(x)GSCTARMAlogMarginal(y,na,nc,q,x);
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
    optim_opts.MaxFunEvals = 5e3;
    problem.options = optim_opts;
    HyperPar = fminsearch(problem);
end

% Computing for the optimized parameter values
options.estim = 'ks';
HP.mu = HyperPar(1:q)';                             % Optimized stochastic constraint vector
HP.theta0 = HyperPar(q+(1:na+nc));                  % Optimized parameter mean
HP.sigma_w2 = exp(HyperPar(end-1));                 % Optimized innovations variance
HP.Sigma_v = exp(HyperPar(end))*eye(na+nc);         % Optimized parameter innovations variance
[theta_hat,P_theta,y_hat,other] = GSCTARMA_MAPtrajectory(y,na,nc,HP,options);

%--------------------------------------------------------------------------
function [logL,theta_hat,P_theta,y_hat,other] = GSCTARMAlogMarginal(y,na,nc,q,HP)

% Obtaining information from the input
Hyperpar.mu = HP(1:q)';
Hyperpar.theta0 = HP(q+(1:na+nc));
Hyperpar.sigma_w2 = exp(HP(end-1));
Hyperpar.Sigma_v = exp(HP(end))*eye(na+nc);

% Making some "security" checks
rho = roots([1 Hyperpar.mu]);
cond1 = min(abs(rho<=1));           % Condition 1 : The roots of the stochastic constraints have magnitude lower than or equal to one
rho = roots([1 Hyperpar.theta0']);
cond2 = min(abs(rho<1));            % Condition 2 : The roots of the mean (initial) parameter vector have magnitude lower than one

% Evaluating the log-likelihood
if cond1 && cond2
    [theta_hat,P_theta,y_hat,other] = GSCTARMA_MAPtrajectory(y,na,nc,Hyperpar);
    logL = other.logL;
else
    logL = inf;
end