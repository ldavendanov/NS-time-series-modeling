function [xtt,Ptt,other] = KalmanFilter(y,u,System,IniCond,Noise)
%--------------------------------------------------------------------------
% Function to compute the Kalman filter estimates of the state of a dynamic
% system based on the signal "y" and the system matrices in "System", using
% the initial conditions and noise properties provided.
% Inputs:
%   y       -> Available measurements --the signal-- ( matrix m x N )
%               m : Size of the measurement vector
%               N : Number of signal samples
%   u       -> Input signal ( matrix n x N )
%               n : Size of the measurement vector
%               N : Number of signal samples
%   System  -> System matrices
%               System.F : State transition matrix ( function of time returning a 2D matrix n x n )
%                          Ex: "System.F = @myfun"
%               System.H : State measurement matrix ( function of time returning a 2D matrix m x n )
%   IniCond -> Initial conditions
%               IniCond.xtt0 : State estimate at t=0 --E[x[0]]-- ( vector n x 1 )
%                              Default value 'IniCond.xtt0 = zeros(n,1)'
%               IniCond.Ptt0 : Covariance of the state estimation error at t=0
%                               --E[ ( x[0] - x[0|0] )( x[0] - x[0|0] )^T ]-- ( matrix n x n )
%                               Default value 'IniCond.Ptt0 = 10^8*eye(n)'    
%   Noise   -> Noise parameters
%               Noise.Q : Process noise covariance matrix ( function of time returning a 2D matrix n x n )
%               Noise.R : Measurement noise covariance matrix ( function of time returning a 2D matrix m x m )
%
% Outputs:
%   xtt     -> Kalman filter estimate of the system state
%   Ptt     -> Estimated state error covariance matrix
%   other   -> Other variables estimated during Kalman filtering
%               other.xttm : A priori estimate of the state vector
%               other.Pttm : A priori estimate of the state error covariance matrix
%               other.K    : Kalman gain
%
% Created by : David Avendaño - August 2013
%
%--------------------------------------------------------------------------

%-- Checking input --------------------------------------------------------
if nargin < 4
    error('KalmanFilter:argChk','Wrong number of input arguments')
end
[m,N] = size(y);                    % Size of the measurement vector
n = size(System.F(1),1);            % Size of the state vector

% Default initial conditions
if isempty(IniCond)
    IniCond.xtt0 = zeros(n,1);
    IniCond.Ptt0 = 1e8*eye(n);
end

%-- Initializing the estimation matrices ----------------------------------
xtt = zeros(n,N);
xtt(:,1) = IniCond.xtt0;
xttm = zeros(n,N);
xttm(:,1) = IniCond.xtt0;
Ptt = zeros(n,n,N);
Ptt(:,:,1) = IniCond.Ptt0;
Pttm = zeros(n,n,N);
Pttm(:,:,1) = IniCond.Ptt0;
K = zeros(n,m,N);
e = zeros(m,N);
ettm = zeros(m,N);
se = zeros(m,m,N);
G = System.G(1);
v = zeros(size(G,2),N);

%-- Kalman filter ---------------------------------------------------------
for i=2:N
    % System matrices
    F = System.F(i); H = System.H(i,e); G = System.G(i);
    R = Noise.R(i);  Q = Noise.Q(i,xtt(:,i-1));
    
    % Update
    if size(u,2) > 0
        xttm(:,i) = F*xtt(:,i-1) + G*u(:,i);
    else
        xttm(:,i) = F*xtt(:,i-1);
    end
    ettm(:,i) = y(:,i) - H*xttm(:,i);
    Pttm(:,:,i) = F*Ptt(:,:,i-1)*F' + G*Q*G';
    
    % Kalman gain
    se(:,i) = H*Pttm(:,:,i)*H' + R;
    K(:,:,i) = Pttm(:,:,i)*H' / se(:,:,i);
    
    % Correction
    xtt(:,i) = xttm(:,i) + K(:,:,i) * ( y(:,i) - H*xttm(:,i)  );
    e(:,i) = y(:,i) - H*xtt(:,i);
    Ptt(:,:,i) = ( eye(n) - K(:,:,i)*H )*Pttm(:,:,i);
    v(:,i) = pinv(G)*( xtt(:,i) - xttm(:,i) );
end

%-- Fixing the output -----------------------------------------------------
other.xttm = xttm;
other.Pttm = Pttm;
other.K = K;
other.e = e;
other.ettm = ettm;
other.se = se;
other.v = v;