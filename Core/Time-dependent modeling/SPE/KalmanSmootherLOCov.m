function [xttN,PttN,PttmN,other] = KalmanSmootherLOCov(y,State,Covariance,System,K,e)
%--------------------------------------------------------------------------
% Function compute the Kalman smoother estimates given the Kalman filter 
% estimates xh[t], using the following SS model
%       x[t] = F(x[t-1]) + G*w[t]
%       y[t] = H(x[t]) + e[t]
% where 
%   -> x[t] is the (n by 1) state vector, 
%   -> F(x[t]) is the (n by 1) state transition function, 
%   -> G (n by l) is the state noise coupling matrix, 
%   -> w[t] is the (l by 1) state noise vector, 
%   -> y[t] is the (m by 1) measurement vector 
%   -> H(x[t]) is the (m by 1) state measurement function
%   -> e[t] is the (m by 1) measurement noise vector
% 
% Inputs 
%   y           ->  Observation vector (m by N)
%   State       ->  Structure with the Kalman filter state estimates (n by N)
%                   State.xtt  : A posteriori estimate of the state
%                   State.xttm : A priori estimate of the state
%   Covariance  ->  Structure with the Kalman filter covariance matrices
%                   Covariance.Ptt  : A posteriori state estimation error covariance matrix
%                   Covariance.Pttm : A priori state estimation error covariance matrix
%   System      -> System matrices
%                   System.F : State transition matrix ( function of time returning a 2D matrix n x n )
%                          Ex: "System.F = @myfun"
%                   System.G : Input matrix matrix ( function of time returning a 2D matrix n x n )
%                          Ex: "System.F = @myfun"
%                   System.H : State measurement matrix ( function of time returning a 2D matrix m x n )
%
% Outputs
%   xtN             ->  Smoothed state vector (n by N)
%   PtN             ->  Smoothed state error covariance matrix (cell array Ph{1 by N}(n by n))
%   other.eN        ->  Smoothed prediction error ( y[t] - H(xhN[t]) ) (m by N)
%
% Ellaborated by: David Avendano Ver 1.0 July 2012
%                 David Avendano Ver 2.0 October 2013
%--------------------------------------------------------------------------

% Retrieving input
xtt = State.xtt;
xttm = State.xttm;
Ptt = Covariance.Ptt;
Pttm = Covariance.Pttm;

% Initializing matrices
[~,N] = size(y);                    % Size of the measurement vector
n = size(System.F(1),1);            % Size of the state vector
xttN = zeros(size(xtt));
PttN = zeros(size(Ptt));
PttmN = zeros(size(Ptt));

% Initial values
xttN(:,N) = xtt(:,N);
PttN(:,:,N) = Ptt(:,:,N);
H = System.H(N,e);
F = System.F(N);
PttmN(:,:,N) = ( eye(n) - K(:,end)*H )*F*Ptt(:,:,N-1);
    
% Computing the smoothed estimates
for t=N-1:-1:1
    % System matrices
    F = System.F(t);
    
    % Smoothing equations
    At = Ptt(:,:,t)*F'*pinv(Pttm(:,:,t+1));
    xttN(:,t) = xtt(:,t) + At*( xttN(:,t+1) - xttm(:,t+1) );
    PttN(:,:,t) = Ptt(:,:,t) + At*( PttN(:,:,t+1) - Pttm(:,:,t+1) )*At';
    
    % Lag-one covariance smoothed estimates
    if t>1
        Atm = Ptt(:,:,t-1)*F'*pinv(Pttm(:,:,t));
    else
        Atm = eye(n);
    end
    PttmN(:,:,t) = Ptt(:,:,t)*Atm' + At*( PttmN(:,:,t+1) - F*Ptt(:,:,t) )*Atm';
    
end


% Obtaining the estimated innovation sequences
ettN = zeros(size(y));
G = System.G(1);
v = zeros(size(G,2),size(xtt,2));
for t = 2:N
    H = System.H(t,ettN);
    G = System.G(t);
    ettN(:,t) = y(:,t) - H*xttN(:,t);
    v(:,t) = G'*( xttN(:,t) - xttm(:,t) );
end

other.ettN = ettN;
other.v = v;