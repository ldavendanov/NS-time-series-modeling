function [theta_hat,P_theta,y_hat,other] = GSCTAR_MAPtrajectory(y,na,Hyperpar,options)
%--------------------------------------------------------------------------
% This function computes the MAP estimates of the parameter trajectories of
% a GSC-TAR$(n_a)$ model based on a time series $y[t]$ and stochastic
% constraint parameters $\mu = [\mu_1 \mu_2 \cdots \mu_q]$, innovations
% variance $\sigma_w^2$ and parameter innovations covariance $\Sigma_v$.
% Input parameters :
%   y        -> Observations vector ( vector : 1 x N )
%   na       -> Model order ( scalar )
%   Hyperpar -> Structure with the model hyperparameters
%       Hyperpar.mu       : Stochastic constraint vector ( vector : 1 x q )
%       Hyperpar.sigma_w2 : Innovations variance ( scalar )
%       Hyperpar.Sigma_v  : Parameter innovations covariance ( matrix : na x na )
%       Hyperpar.theta0   : Parameter mean ( matrix : na x 1 )
%   options -> Structure with estimator options
%       options.estim : Estimator type ( Kalman filter : 'kf' (default), or Kalman smoother : 'ks' )
%
% Output parameters:
%   theta_hat -> Estimated parameter trajectories ( matrix : na x N )
%   P_theta   -> Parameter estimation error covariance ( 3D matrix : na x na x N )
%   y_hat     -> Estimated signal structure
%       y_hat.pred   : One-step-ahead prediction error ( vector : 1 x N )
%       y_hat.filt   : Filtering error ( vector : 1 x N )
%       y_hat.smooth : Smoothing error ( vector : 1 x N )
%
% Created by : David Avendano - September 2015
%--------------------------------------------------------------------------

% Declaring global scope variables
global N naG q muG sw Sv yG
N = length(y);
yG = y;
naG = na;                               % AR order
muG = Hyperpar.mu;                      % Stochastic constraint vector
sw = Hyperpar.sigma_w2;                 % Innovations variance
Sv = Hyperpar.Sigma_v;                  % Parameter innovations variance
q = length(muG);                        % Stochastic constraint order

if nargin == 3
    options.estim = 'kf';
end

% Setting up the state space representation
System.F = @stm; System.H = @smm; System.G = @ncm;  % State space matrices
Noise.R = @InnovVar; Noise.Q = @ParamInnovVar;      % Noise matrices
IniCond.xtt0 = kron(ones(q,1),Hyperpar.theta0);     % Initial conditions
IniCond.Ptt0 = 1e-5*eye(na*q);

% Check for unit roots in the stochastic constraint parameters
u = [];
rho = roots([1 muG]);
if abs(rho) ~= 1,
    v0 = Hyperpar.theta0*sum([1 muG]);
    u = kron(v0,ones(1,N));
else
    v0 = zeros(na,1);
end

% Computing Kalman filter estimates
[xtt,Ptt,other] = KalmanFilter(y,u,System,IniCond,Noise);

% Extracting the results from one-step-ahead prediction
theta_hat.pred = other.xttm(1:na,:);                    % One-step-ahead prediction of the parameter vector
P_theta.pred = other.Pttm(1:na,1:na,:);                 % Covariance matrix of the parameter prediction error
y_hat.pred = y+other.ettm;                              % One-step-ahead signal prediction

% Extracting the results from filtering
theta_hat.filt = xtt(1:na,:);                           % Filtered estimate of the parameter vector
P_theta.filt = Ptt(1:na,1:na,:);                        % Covariance matrix of the filtered parameter estimate
y_hat.filt = y+other.e;                                 % Filtered estimate of the signal

% Evaluating the marginal likelihood
T = na+1:N-na;                                          % Evaluation period
ettm = other.ettm(T);                                   % One-step-ahead prediction error
se = squeeze(other.se(:,:,T))';                         % Variance of the one-step-ahead prediction error
logL = sum(log(se))/2 + sum( ettm.^2./se )/2;           % Log Marginal Likelihood
other.logL = logL;

% Evaluating the joint likelihood
ett = other.e(T);
J = -(1/2)*sum( log(sw) + ett.^2/sw );
v = xtt(1:na,T) - other.xttm(1:na,T);
J = J - (length(T)*log(det(Sv))/2) - (1/2)*trace( (v'/Sv)*v );
other.J = J;

% Computing smoothed estimates
if strcmp(options.estim,'ks')
    State.xtt = xtt;
    State.xttm = other.xttm;
    Covariance.Ptt = Ptt;
    Covariance.Pttm = other.Pttm;
    [xttN,PttN,otherS] = KalmanSmoother(y,State,Covariance,System);
    
    % Extracting the results from smoothing
    theta_hat.smooth = xttN(1:na,:);                    % Smoothed estimates of the parameter vector
    P_theta.smooth = PttN(1:na,1:na,:);                 % Covariance matrix of the smoothed parameter estimates
    y_hat.smooth = y+otherS.ettN;                       % Filtered estimate of the signal

elseif strcmp(options.estim,'kc')
    State.xtt = xtt;
    State.xttm = other.xttm;
    Covariance.Ptt = Ptt;
    Covariance.Pttm = other.Pttm;
    [xttN,PttN,PttmN,otherS] = KalmanSmootherLOCov(y,State,Covariance,System,other.K);
    
    % Extracting the results from smoothing
    theta_hat.smooth = xttN(1:na,:);                    % Smoothed estimates of the parameter vector
    P_theta.smooth = PttN(1:na,1:na,:);                 % Covariance matrix of the smoothed parameter estimates
    y_hat.smooth = y+otherS.ettN;                       % Filtered estimate of the signal 
    
    % Computing the expected negative log-likelihood
    C1 = 0; C2 = 0;
    G = System.G(1);
    F = System.F(1);
    for t=T
        h = System.H(t,:);        
        C1 = C1 + otherS.ettN(t).^2 + h*PttN(:,:,t)*h';
        C2 = C2 + ( ( xttN(:,t) - F*xttN(:,t-1) )'*G/Noise.Q(1) )*v0;
    end
    
    S11 = xttN(:,T)*xttN(:,T)' + sum(PttN(:,:,T),3);
    S00 = xttN(:,T-1)*xttN(:,T-1)' + sum(PttN(:,:,T-1),3);
    S10 = xttN(:,T)*xttN(:,T-1)' + sum(PttmN(:,:,T),3);
    Svz = (G*v0)*sum(xttN(:,T),2)';
    
    C2 = C2 + (N-2*na)*(v0'/Noise.Q(1))*v0;
    C0 = (N-2*na)*( log(det(Noise.Q(1))) + log(Noise.R(1)) );
    Q = C0 + C1/(Noise.R(1)) + C2 + trace( Noise.Q(1) \ G'*( S11 - F*S10' - S10*F' + F*S00*F' )*G );
    
    % Providing other output variables
    other.Q = Q;
    other.S11 = S11;
    other.S00 = S00;
    other.S10 = S10;
    other.Svz = Svz;
    other.C = C1;
end

%--------------------------------------------------------------------------
function F = stm(~)
global q muG naG
F = [-muG; eye(q-1,q)];
F = kron(F,eye(naG));

%--------------------------------------------------------------------------
function H = smm(t,~)
global naG q yG
max_del = min(naG,t-1);
H = zeros(1,naG*q);
H(1:max_del) = -yG(t-1:-1:t-max_del);

%--------------------------------------------------------------------------
function G = ncm(~)
global naG q
G = eye(q,1);
G = kron(G,eye(naG));

%--------------------------------------------------------------------------
function R = InnovVar(~)
global sw
R = sw;

%--------------------------------------------------------------------------
function Q = ParamInnovVar(~,~)
global Sv
Q = Sv;