function [xtt,Ptt,other] = ExtendedKalmanFilter(y,System,IniCond,Noise)
%--------------------------------------------------------------------------
% Function to compute the Extended Kalman filter estimates of the state of
% a dynamic system based on the signal "y" and the system matrices in
% "System", using the initial conditions and noise properties provided.
% Inputs:
%   y       -> Available measurements --the signal-- ( matrix m x N )
%               m : Size of the measurement vector
%               N : Number of signal samples
%   System  -> System matrices
%               System.F    : Linearized state transition matrix ( 2D matrix n x n --LTI system-- )
%                                                                ( 3D matrix n x n x N --LTV system-- )
%               System.ffun : State transition function ( function handle Ex: ffun = @MyFFun )
%               System.H    : Linearized state measurement matrix ( 2D matrix m x n --LTI system-- )
%                                                                 ( 3D matrix m x n x N --LTV system-- )
%               System.hfun : Measurement function ( function handle Ex: hfun = @MyHFun )
%
%   IniCond -> Initial conditions
%               IniCond.xtt0 : State estimate at t=0 --E[x[0]]-- ( vector n x 1 )
%                              Default value 'IniCond.xtt0 = zeros(n,1)'
%               IniCond.Ptt0 : Covariance of the state estimation error at t=0
%                               --E[ ( x[0] - x[0|0] )( x[0] - x[0|0] )^T ]-- ( matrix n x n )
%                               Default value 'IniCond.Ptt0 = 10^8*eye(n)'    
%   Noise   -> Noise parameters
%               Noise.Q : Process noise covariance matrix ( 2D matrix n x n --homoskedastic system-- )
%                                                         ( 3D matrix n x n x N --heteroskedastic system-- )
%               Noise.R : Measurement noise covariance matrix ( 2D matrix m x m --homoskedastic system-- )
%                                                             ( 3D matrix m x m x N --heteroskedastic system-- )
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
n = size(Noise.Q([],[]),1);         % Size of the state vector

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
v = zeros(n,N);

%-- Kalman filter ---------------------------------------------------------
for i=2:N
    F = System.F(xtt(:,i-1),i); H = System.H(i,e);
    R = Noise.R(i);  Q = Noise.Q(i,xtt(:,i-1));
    
    xttm(:,i) = System.ffun( xtt(:,i-1), i );
    Pttm(:,:,i) = F*Ptt(:,:,i-1)*F' + Q;
    K(:,:,i) = Pttm(:,:,i)*H' / ( H*Pttm(:,:,i)*H' + R );
    xtt(:,i) = xttm(:,i) + K(:,:,i) * ( y(:,i) - System.hfun( xttm(:,i), i, e ) );
    Ptt(:,:,i) = ( eye(n) - K(:,:,i)*H )*Pttm(:,:,i);
    e(i) = y(:,i) - feval( System.hfun, xtt(:,i), i, e );
    v(:,i) = xtt(:,i) - xttm(:,i);
end

%-- Fixing the output -----------------------------------------------------
other.xttm = xttm;
other.Pttm = Pttm;
other.K = K;
other.e = e;
other.v = v;