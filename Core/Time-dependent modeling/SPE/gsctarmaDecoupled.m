function [a,c,se,other] = gsctarmaDecoupled(obs,order,options)
%--------------------------------------------------------------------------
% Function to estimate a GSC-TARMA model parameters and hyperparameters by
% means of the decoupled method and Maximum Likelihood objective function
% Inputs:
%   y       ->  (m by N) measurement vector
%   order 	->  [na,nc,kappa] AR and MA polynomial orders and stochastic
%                constraint order
%   options ->  Estimating options
%   options.P               : Hyperparameters of the GSC-TARMA model
%   options.P.sigma_w       : Initial value of the signal innovations variance
%   options.P.Sigma_w       : Initial value of the parameter innovations covariance matrix
%   options.P.mu            : Initial value of the stochastic constraint parameters
%   options.IniCond.a0      : Initial value of the time-dependent AR coefficient vector
%   options.IniCond.c0      : Initial value of the time-dependent MA coefficient vector
%   options.IniCond.Ptt0    : Initial value of the covariance matrix of the state estimation error 
%   options.VarianceEstimator.Type : Type of estimator for the TV innovations variance
%                                       'c'     ->  Constant variance
%                                       'np'    ->  Non-parametric moving window
%   options.VarianceEstimator.Parameters : Parameters of the innovations variance estimator:
%                                           -> Non-parametric ('np') = nwin (window size) 
% Outputs:
%   a       ->  (na+1 by N) estimate of TV-AR parameter vector (includes a_0[t] = 1)
%   c       ->  (nc+1 by N) estimate of TV-AR parameter vector (includes c_0[t] = 1)
%   se      ->  (m by N) estimate of the TV innovations variance
%   other   ->  Structure with estimation and performance summary
%
% Ellaborated by:   David Avendano, October 2012, Ver 2.0
%                   David Avendano, December 2012, Ver 3.0 (non linear estimator of the GSC-TARMA hyperparameters)
%                   David Avendano, February 2013, Ver 3.5 (revision + model validation)
%                   David Avendano, October 2013, Ver 4
%--------------------------------------------------------------------------

% Definition of global variables
global y na nc q Sigma_v sigma_w ini
y = obs;

%-- Checking input --------------------------------------------------------
if nargin < 2
    error('gsctarmaDecoupled:argChk','Wrong number of input arguments')
end

% Setting the model structure
switch numel(order)
    case 1 % GSC-TAR model with na provided by the user and q = 1
        na = order(1);
        nc = 0;
        q = 1;
    case 2  % GSC-TAR model with na and q provided by the user
        na = order(1);
        nc = 0;
        q = order(2);
    case 3 % GSC-TARMA model with na, nc and q provided by the user
        na = order(1);
        nc = order(2);
        q = order(3);
end

%-- Default options -------------------------------------------------------
if nargin < 3
    options.P = DefaultHyperparams;
    options.VarianceEstimator.Type = 'c';
    options.IniCond = DefaultIniCond;
end
if ~isfield(options.IniCond,'c0')
    options.IniCond.c0 = zeros(nc,1);
end

if ~isfield(options,'P')
    options.P = DefaultHyperparams;
end
if ~isfield(options,'IniCond')
    options.IniCond = DefaultInitCond;
end
if ~isfield(options,'VarianceEstimator')
    options.VarianceEstimator.Type = 'c';
end
if ~isfield(options,'gamma')
    options.gamma = 5e-1;

end
if ~isfield(options,'ShowProgress')
    options.ShowProgress = 1;
end

%-- Setting up the state space representation matrices for the KF ---------
Sigma_v = options.P.Sigma_v;                    % Parameter innovations covariance
sigma_w = options.P.sigma_w;                    % Innovations variance
mu = zeros(q,options.max_iter);                 % Stochastic constraint parameters
theta0 = zeros(na+nc,options.max_iter);         % Mean of the TARMA parameter vector
mu(:,1) = options.P.mu;                         
theta0(:,1) = [options.IniCond.a0; options.IniCond.c0];

%-- Optimization properties -----------------------------------------------
gamma = options.gamma;
Gamma = diag([gamma*ones(1,q) 1e-6*gamma*ones(1,na+nc)]);
ini = 100;                                      % Initial number of samples for averaging

%-- Setting up the initial conditions for the KF --------------------------
options.IniCond.xtt0 = zeros((na+nc)*q,1);
options.IniCond.Ptt0 = 1e8*eye((na+nc)*q);
    
%-- Initialization of estimation matrices ---------------------------------
Iterations.RSS_SSS = zeros(1,options.max_iter);
Iterations.PESS = zeros(1,options.max_iter);
Iterations.ObjFun = zeros(1,options.max_iter);

%-- Estimation with the Newton optimization method ------------------------
if options.ShowProgress == 1,
    fprintf('Iter  - Objective - dObjective -   |phi|   -   |d_phi|  \n')
end
for iter = 2:options.max_iter
    
    % Estimating the GSC-TARMA parameters and computing the gradients
    [Jacob,Hess,f,xtt,other] = GSCjacobian(mu(:,iter-1)',theta0(:,iter-1),options);
    
    % Newton's method update rule
    phi_ = [mu(:,iter-1); theta0(:,iter-1)];
    
    phi = phi_ - Gamma*(pinv(Hess)*Jacob);
    mu(:,iter) = phi(1:q);
    theta0(:,iter) = phi(q+1:end); 
    
    % Updating initial conditions
    options.IniCond.xtt0 = xtt(:,end);
    options.IniCond.Ptt0 = 10^(-1)*options.IniCond.Ptt0;
    
    % Displaying performance
    dObjFun = ( f - Iterations.ObjFun(iter-1) ) / Iterations.ObjFun(iter-1);
    d_mu = norm( abs( phi - phi_ ) / phi_ );
    if options.ShowProgress == 1,
        fprintf('  %3d - %2.3e - %2.3e - %2.3e - %2.3e \n', iter,...
            f, dObjFun, norm(phi), d_mu )
    end
    Iterations.RSS_SSS(iter) = other.Performance.rss_sss;
    Iterations.PESS(iter) = other.Performance.pess;
    Iterations.ObjFun(iter) = f;
    
    if abs(dObjFun) < options.TolFun || abs(d_mu) < 1e-2*options.TolFun
        break
    end
    
end

% Organizing the output
Iterations.max_iter = iter;
Iterations.RSS_SSS = Iterations.RSS_SSS(1:iter);
Iterations.PESS = Iterations.PESS(1:iter);
Iterations.ObjFun = Iterations.ObjFun(1:iter);
Iterations.Mu = mu(:,1:iter);
Iterations.theta0 = theta0(:,1:iter);
other.Iterations = Iterations;
a = xtt(1:na,:) + repmat(theta0(1:na,iter),1,length(y));
c = xtt(na+1:na+nc,:) + repmat(theta0(na+1:na+nc,iter),1,length(y));
se = var(other.e);

%--------------------------------------------------------------------------
function [Jacob,Hess,f,xtt,other] = GSCjacobian(mu,theta0,options)
global na nc q ini

%- Estimating the GSC-TARMA parameters with the Kalman filter
[xtt,other] = KFestimGSC(mu,theta0,options);
N = size(xtt,2);

%- Estimating the innovations variance and parameter innovations covariance matrix
Sigma_v = StateNoise;
sigma_w = MeasNoise;

%- Initializing computing matrices
Jacob = zeros(q+na+nc,N);
Hess = zeros(q+na+nc,q+na+nc,N);
f = zeros(N,1);
dx_dmu_ = zeros((na+nc)*q,q);

% Computing the instantaneous values of the Jacobian and the Hessian
for t=2:N
    
    % Measurement matrix
    F = stm( t );
    H = smm( t, other.e );
    
    % Objective function
    f(t) = (1/2)*( other.e(t).^2/sigma_w + other.v(:,t)'*pinv(Sigma_v)*other.v(:,t) );
    
    % Computing the gradients with respect to the stochastic constraint parameters
    theta = xtt(:,t-1) - repmat(theta0,q,1);
    dxm_dmu = zeros((na+nc)*q,q);
    dxm_dmu(1:na+nc,:) = -reshape(theta,na+nc,q);
    dxm_dmu = dxm_dmu + F*dx_dmu_;
    dx_dmu = ( eye((na+nc)*q) - other.K(:,t)*H )*dxm_dmu;
    dx_dmu_ = dx_dmu;
    
    % Computing the instantaneous Jacobian and Hessian matrices
    dx_dpar = dx_dmu;
    dxm_dpar = dxm_dmu;
    Jacob(1:q,t) = (-other.e(t)/sigma_w*H*dx_dpar + other.v(:,t)'*pinv(Sigma_v)*( dx_dpar - dxm_dpar ))';
    Jacob(q+1:end,t) = -H(1:na+nc)*other.e(t)/sigma_w;
    Hess(1:q,1:q,t) = (1/sigma_w)*dx_dpar'*(H')*H*dx_dpar + ( dx_dpar - dxm_dpar )'*pinv(Sigma_v)*( dx_dpar - dxm_dpar );
    Hess(q+1:end,q+1:end,t) = H(1:na+nc)*H(1:na+nc)'/sigma_w;
%     Hess(q+1:end,1:q,t) = H(1:na+nc)'*(H*dx_dpar)/sigma_w;
%     Hess(1:q,q+1:end,t) = Hess(q+1:end,1:q,t)';
end

% Overall values of the Jacobian, the Hessian and the objective function
Jacob = sum(Jacob(:,ini:end),2);
Hess = sum(Hess(:,:,ini:end),3);
f = sum(f);

%--------------------------------------------------------------------------
function [xtt,other] = KFestimGSC(mu,theta0,options)
global y q Sigma_v sigma_w P

%- Setting up the state space representation matrices for the KF
System.H = @smm;                                % System's measurement matrix
System.F = @stm;                                % System's state transition matrix
Noise.Q = @StateNoise;                          % State noise matrix
Noise.R = @MeasNoise;                           % Measurement noise matrix

%- Estimate the TARMA parameter vector with the Kalman filter for given values of P
P.mu = mu;
P.Sigma_v = Sigma_v;
P.sigma_w = sigma_w;
u = repmat(theta0,q,length(y));
[xtt,Ptt,other] = KalmanFilter(y,u,System,options.IniCond,Noise);

%- Evaluate the performance
Performance = PerfMeasures(xtt,Ptt,other);

%- Organizing the output
other.P = P;
other.Performance = Performance;
other.IniCond = options.IniCond;
other.Ptt = Ptt;

%--------------------------------------------------------------------------
function F = stm(~)
% Function to compute the state transition matrix of the GSC-TARMA model
global na nc q P
F = [-P.mu; eye(q-1,q)];
F = kron(F,eye(na+nc));

%--------------------------------------------------------------------------
function H = smm( t, e )
% Function to compute the measurement matrix of the GSC-TARMA model
global y na nc q

mx_dely = min(na,t-1);
mx_dele = min(nc,t-1);

H = zeros(1,(na+nc)*q);
H(1:mx_dely) = -y(t-1:-1:t-mx_dely);
H(na+1:na+mx_dele) = e(t-1:-1:t-mx_dele);

%--------------------------------------------------------------------------
function R = MeasNoise(~)
% Function to generate the signal innovations variance of the GSC-TARMA model
global P
R = P.sigma_w;

%--------------------------------------------------------------------------
function Q = StateNoise(~,~)
% Function to compute the parameter innovations covariance matrix of the
% GSC-TARMA model

global na nc q P
G = eye((na+nc)*q,na+nc);
Q = G*P.Sigma_v*G';

%--------------------------------------------------------------------------
function Performance = PerfMeasures(xtt,Ptt,other)
global na nc q y ini

% Size of the matrices
N = size(xtt,2);
M = (na+nc)*q;

%- Retrieving the innovations variance and parameter innovations covariance matrix
Sigma_v = StateNoise;
sigma_w = MeasNoise;

% Computing the performance measures
Performance.rss = sum(other.e(ini:end).^2);
Performance.rss_sss = Performance.rss/sum(y(ini:end).^2);
Performance.pess = trace(other.v(:,ini:end)*other.v(:,ini:end)');
D = (other.e(ini:end).^2./sigma_w) + diag(other.v(:,ini:end)'*pinv(Sigma_v)*other.v(:,ini:end))';
Performance.JG = -(N*(M+1)/2)*log(2*pi) - (N/2)*log(det(Sigma_v)) - (N/2)*log(sigma_w) - (1/2)*sum(D);
Performance.ObjFun = (1/2)*sum(D);

CN_Ptt = zeros(1,N);
for i=1:N
    CN_Ptt = cond(Ptt(:,:,i));
end
Performance.CN_Ptt = CN_Ptt;

%--------------------------------------------------------------------------
function P = DefaultHyperparams
global na nc q

P.sigma_w = 1;
P.Sigma_v = 1e-2*eye(na+nc);
P.mu = poly(ones(q,1));
P.mu = P.mu(2:end);

%--------------------------------------------------------------------------
function IniCond = DefaultInitCond
global na nc q

IniCond.a0 = zeros(na,1);
IniCond.c0 = zeros(nc,1);
IniCond.Ptt0 = 1e8*eye((na+nc)*q);
