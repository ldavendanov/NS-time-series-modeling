function [a,c,se,other] = gsctarmaML(obs,order,options)
%--------------------------------------------------------------------------
% Function to estimate a GSC-TARMA model parameters and hyperparameters by
% means of the Maximum Likelihood method
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
global y na nc q Sigma_v sigma_w
y = obs;

%-- Checking input --------------------------------------------------------
if nargin < 2
    error('gsctarmaML:argChk','Wrong number of input arguments')
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

if ~isfield(options,'ShowProgress')
    options.ShowProgress = 1;
end

%-- Setting up the state space representation matrices for the KF ---------
Sigma_v = options.P.Sigma_v;                    % Parameter innovations covariance
sigma_w = options.P.sigma_w;                    % Innovations variance
mu0 = options.P.mu;                             % Stochastic constraint parameters

%-- Estimation with the ML method
opts = optimset('fminsearch');
opts.Display = 'iter';
opts.MaxIter = 200;
mu = fminsearch(@(mu)KFestimGSC(mu,options),mu0,opts);
[~,xtt,other] = KFestimGSC(mu,options);

% Organizing the output
a = xtt(1:na,:);
c = xtt(na+1:na+nc,:);
se = var(other.e);

%--------------------------------------------------------------------------
function [obj,xtt,other] = KFestimGSC(mu,options)
global y na nc q Sigma_v sigma_w P

%-- Setting up the state space representation matrices for the KF ---------
System.H = @smm;                                % System's measurement matrix
System.F = @stm;                                % System's state transition matrix
System.G = @im;                                 % System's input matrix
Noise.Q = @StateNoise;                          % State noise matrix
Noise.R = @MeasNoise;                           % Measurement noise matrix

%-- Initial conditions ----------------------------------------------------
IniCond.xtt0 = zeros((na+nc)*q,1);              % Initial values of the state vector
IniCond.Ptt0 = options.IniCond.Ptt0;            % Initial state estimation error covariance
theta0 = [options.IniCond.a0; options.IniCond.c0];

%- Estimate the TARMA parameter vector with the Kalman filter for given values of P
P.mu = mu;
P.Sigma_v = Sigma_v;
P.sigma_w = sigma_w;
u = repmat(theta0,q,length(y));
[xtt,Ptt,other] = KalmanFilter(y,u,System,IniCond,Noise);

%- Evaluate the performance
Performance = PerfMeasures(xtt,Ptt,other);
obj = -Performance.JG;

% Organizing the output
other.P = P;
other.Performance = Performance;
other.IniCond = IniCond;
other.Ptt = Ptt;

%--------------------------------------------------------------------------
function F = stm(~)
% Function to compute the state transition matrix of the GSC-TARMA model
global na nc q P
F = [-P.mu; eye(q-1,q)];
F = kron(F,eye(na+nc));

%--------------------------------------------------------------------------
function G = im(~)
% Function to compute the input matrix of the GSC-TARMA model
F = stm;
G = eye(size(F)) - F;

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
global na nc q y

ini = 100;
N = size(xtt,2);
M = (na+nc)*q;
Sigma_v = other.v(:,ini:end)*other.v(:,ini:end)';
sigma_w = var(other.e(ini:end))*ones(1,N);
Performance.rss = sum(other.e(ini:end).^2);
Performance.rss_sss = Performance.rss/sum(y(ini:end).^2);
Performance.pess = trace(Sigma_v);
D = log(sigma_w(ini:end)) + (other.e(ini:end).^2./sigma_w(ini:end)) + diag(other.v(:,ini:end)'*Sigma_v^(-1)*other.v(:,ini:end))';
Performance.JG = -(N*(M+1)/2)*log(2*pi) - (N/2)*log(det(Sigma_v)) - (1/2)*sum(D);

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
