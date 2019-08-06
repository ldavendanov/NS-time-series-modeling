function [a,c,se,other] = sptarma(obs,order,options)
%--------------------------------------------------------------------------
% Function to estimate an SP-TARMA model parameters using the EKF
% Inputs:
%   y       ->  (m by N) measurement vector
%   order 	->  [na,nc,kappa] AR and MA polynomial orders and stochastic
%                constraint order
%   options ->  Estimating options
%   options.P               : Hyperparameters of the GSC-TARMA model
%   options.P.sigma_w       : Signal innovations variance
%   options.P.Sigma_v       : Parameter innovations covariance matrix
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
% Ellaborated by:   David Avendano, February 2014
%--------------------------------------------------------------------------

% Definition of global variables
global y na nc q ini
y = obs;
N = length(y);

%-- Checking input --------------------------------------------------------
if nargin < 2
    error('sptarma:argChk','Wrong number of input arguments')
end

% Setting the model structure
switch numel(order)
    case 1 % SP-TAR model with na provided by the user and q = 1
        na = order(1);
        nc = 0;
        q = 1;
    case 2  % SP-TAR model with na and q provided by the user
        na = order(1);
        nc = 0;
        q = order(2);
    case 3 % SP-TARMA model with na, nc and q provided by the user
        na = order(1);
        nc = order(2);
        q = order(3);
end

%-- Default options -------------------------------------------------------
%- Case 1: The user provides only the signal and orders
if nargin < 3
    options.P = DefaultHyperparams;
    options.VarianceEstimator.Type = 'np';
    options.VarianceEstimator.Param = 200;
    options.IniCond = DefaultIniCond;
end
%- Case 2: The user provides all the input parameters but the options field
%          is incomplete  
%- Case 2.1: Providing default initial conditions
if ~isfield(options,'IniCond')
    options.IniCond = DefaultIniCond;
end
%- Case 2.2: Default values for the MA parameters at time 0
if ~isfield(options.IniCond,'c0')
    options.IniCond.c0 = zeros(nc,1);
end
%- Case 2.3: Providing default values for the hyperparameters
if ~isfield(options,'P')
    options.P = DefaultHyperparams;
end
%- Case 2.4: Providing default variance estimator
if ~isfield(options,'VarianceEstimator')
    options.VarianceEstimator.Type = 'c';
end
%- Case 2.5: By default the algorithm shows the achieved performance
if ~isfield(options,'ShowProgress')
    options.ShowProgress = 1;
end
%- Case 2.6: Providing the default transient period (samples) to be removed 
%            from the performance computation
if ~isfield(options,'InitialSamples')
    ini = 100;
else
    ini = options.InitialSamples;
end

%-- Setting up the initial conditions for the KF --------------------------
options.IniCond.xtt0 = zeros((na+nc)*q,1);
options.IniCond.xtt0(1:na+nc) = [options.IniCond.a0; options.IniCond.c0];
    
%-- Obtaining the estimates for the optimal set point ---------------------
if nc == 0
    fprintf('Estimating an SP-TAR( %2d ) q = %2d model\n',na,q)
else
    fprintf('Estimating an SP-TARMA( %2d, %2d ) q = %2d model\n',na,nc,q)
end
[xttN,other] = KFestimSP(options);

fprintf('Performance: ObjFun = %2.2e, RSS/SSS = %2.2e\n',...
    other.Performance.ObjFun,other.Performance.rss_sss)

a = xttN(1:na,:);
c = xttN(na+1:na+nc,:);
switch options.VarianceEstimator.Type  
    case 'c'
        se = var(other.e);
    case 'np'
        se = zeros(1,N);
        buff = zeros(1,options.VarianceEstimator.Param);
        for tt = 1:N
            buff = [other.e(tt) buff(1:end-1)];
            se(tt) = mean(buff.^2);
        end
end

%--------------------------------------------------------------------------
function [xttN,other] = KFestimSP(options)
global y q P na nc

%- Setting up the state space representation matrices for the KF
System.H = @smm;                                % System's measurement matrix
System.F = @stm;                                % System's state transition matrix
System.G = @ncm;                                % System's noise coupling matrix
Noise.Q = @StateNoise;                          % State noise matrix
Noise.R = @MeasNoise;                           % Measurement noise matrix

%- Estimate the TARMA parameter vector with the Kalman filter for given values of P
disp('Forward Kalman filter')
P = options.P;
if ~isfield(P,'mu')
    mu = poly(ones(q,1));
    P.mu = mu(2:end);
end
if ~isfield(P,'vo')
    u = zeros(na+nc,length(y));
else
    u = repmat(P.vo',1,length(y));
end
[xtt,Ptt,oth] = KalmanFilter(y,u,System,options.IniCond,Noise);

%- Obtaining smoothed estimates using the Kalman smoother
disp('Backwards smoothing')
State.xtt = xtt;
State.xttm = oth.xttm;
Covariance.Ptt = Ptt;
Covariance.Pttm = oth.Pttm;
[xttN,PttN,other] = KalmanSmoother(y,State,Covariance,System);

%- Evaluate the performance
Performance = PerfMeasures(xtt,Ptt,other);

%- Organizing the output
other.P = P;
other.Performance = Performance;
other.IniCond = options.IniCond;
other.xtt = xtt;
other.Ptt = Ptt;
other.PttN = PttN;
other.ett = oth.e;
other.ettm = oth.ettm;
other.logMarginal = oth.logMarginal;

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
function G = ncm(~,~)
% Function to compute the process noise coupling matrix of the GSC-TARMA model
global na nc q
G = eye((na+nc)*q,na+nc);

%--------------------------------------------------------------------------
function R = MeasNoise(~)
% Function to generate the signal innovations variance of the GSC-TARMA model
global P
R = P.sigma_w;

%--------------------------------------------------------------------------
function Q = StateNoise(~,~)
% Function to compute the parameter innovations covariance matrix of the
% GSC-TARMA model

global P
Q = P.Sigma_v;

%--------------------------------------------------------------------------
function Performance = PerfMeasures(xtt,Ptt,other)
global na nc q y ini P

% Size of the matrices
N = size(xtt,2);
M = (na+nc)*q;

%- Retrieving the innovations variance and parameter innovations covariance matrix
Sigma_v = P.Sigma_v;
sigma_w = P.sigma_w;

% Computing the performance measures
Performance.rss = sum(other.e(ini:end).^2);
Performance.rss_sss = Performance.rss/sum(y(ini:end).^2);
Performance.pess = trace(other.v(:,ini:end)*other.v(:,ini:end)');
D = (other.e(ini:end).^2./sigma_w) + diag((other.v(:,ini:end)'/Sigma_v)*other.v(:,ini:end))';
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
P.Sigma_v = 1e-3*eye(na+nc);
P.mu = poly(ones(q,1));
P.mu = P.mu(2:end);

%--------------------------------------------------------------------------
function IniCond = DefaultIniCond
global na nc q

IniCond.a0 = zeros(na,1);
IniCond.c0 = zeros(nc,1);
IniCond.Ptt0 = 1e8*eye((na+nc)*q);
