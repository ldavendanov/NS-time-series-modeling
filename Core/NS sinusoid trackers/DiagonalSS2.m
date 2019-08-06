function [ym,Am,omega,theta,error,logMarginal] = DiagonalSS2(y,M,variances)
%--------------------------------------------------------------------------
% Estimation of the modal components of a non-stationary signal using the
% Dual EKF method 
% Created by : David Avendano - April 2016
%--------------------------------------------------------------------------

n = 2*M;            % Dimension of the modal and parameter vector
N = length(y);      % Lenght of the signal

% Setting up the state space representation
System.F = @stm;
System.H = @smm;
System.Ctheta = @pmm;

% Setting up the noise features
R = 1;                  % Measurement noise variance
if nargin < 3
    lambda = 1e-4;
    mu = 5e2*lambda;
else
    R = variances(1);
    lambda = variances(2);      % Variance of the parameters
    mu = variances(3);          % Variance of the mode trajectories
end
Q = mu*eye(n);          % State noise covariance matrix
Sv = lambda*eye(n);     % Parameter noise covariance matrix

% Initializing the estimation matrices
xtt = zeros(n,N);   xttm = zeros(n,N);
Ptt = zeros(n,n,N); Pttm = zeros(n,n,N);
K = zeros(n,1,N);
th_tt = zeros(n,N); th_ttm = zeros(n,N);
Pth_tt = zeros(n,n,N); Pth_ttm = zeros(n,n,N);
Kth = zeros(n,1,N);
e = zeros(1,N);
ettm = zeros(1,N);
logMarginal = zeros(1,N);

% Initialization
a = arburg(y(1:50*M),2*M);
A = compan(a);
[~,D,V] = eig(A);
alpha = diag(D);
Wo = [1 1; 1i -1i];
W = kron(eye(M),Wo);
z0 = y(n:-1:1);
z0 = z0(:);

% Setting up the initial modal values
xtt(1:n,1) = real((V/W)*z0);
Ptt(:,:,1) = 1e-4*eye(n);
Pttm(:,:,1) = Ptt(:,:,1);

% Setting up the initial parameter values
for m=1:M
    th_tt(2*m-1,1) = real(alpha(2*m-1));
    th_tt(2*m,1)   = imag(alpha(2*m-1));
end
th_ttm(:,1) = th_tt(:,1);
Pth_tt(:,:,1) = 1e-4*eye(n); 
Pth_ttm(:,:,1) = Pth_tt(:,:,1);

%-- Kalman filter ---------------------------------------------------------
H = System.H(M);
for i=2:N    
    % Parameter prediction equations
    th_ttm(:,i) = th_tt(:,i-1);
    Pth_ttm(:,:,i) = Pth_tt(:,:,i-1) + Sv;
    
    % State prediction equations
    F = System.F(th_ttm(:,i)); 
    xttm(:,i) = F*xtt(:,i-1);
    Pttm(:,:,i) = F*Ptt(:,:,i-1)*F' + Q;
    
    % Prediction error
    ettm(:,i) = y(:,i) - H*xttm(:,i);
    sigma_e2 = H*Pttm(:,:,i)*H' + R;
    logMarginal(i) = -0.5*(log(sigma_e2) + ettm(:,i).^2/sigma_e2);
    
    % State update equations
    K(:,:,i) = Pttm(:,:,i)*H' / sigma_e2;
    xtt(:,i) = xttm(:,i) + K(:,:,i) * ettm(:,i);
    Ptt(:,:,i) = ( eye(n) - K(:,:,i)*H )*Pttm(:,:,i);
    
    % Parameter update equations
    Cth = System.Ctheta(xtt(:,i-1));
    Kth(:,:,i) = Pth_ttm(:,:,i)*Cth' / ( Cth*Pth_ttm(:,:,i)*Cth' + R );
    th_tt(:,i) = th_ttm(:,i) + Kth(:,:,i) * ettm(:,i);
    Pth_tt(:,:,i) = ( eye(n) - Kth(:,:,i)*Cth )*Pth_ttm(:,:,i);
    
    % Update error
    e(:,i) = y(:,i) - H*xtt(:,i);
    
end

% Yielding resulting modal decomposition
ym = xtt(1:2:n,:);
theta = th_tt(1:n,:);
omega = zeros(M,N);
Am = zeros(M,N);
for m=1:M
    omega(m,:) = atan2(theta(2*m,:),theta(2*m-1,:));
    Am(m,:) = sqrt( xtt(2*m-1,:).^2 + xtt(2*m,:).^2 );
end

% Sorting the modes from lower to higher frequency
mn_omega = mean(omega,2);
[~,ind] = sort(mn_omega);
ym = ym(ind,:);
omega = omega(ind,:);
Am = Am(ind,:);

% Error time series
error.prior = ettm;
error.posterior = e;
T = 201:N;
logMarginal = sum(logMarginal(T));

%--------------------------------------------------------------------------
function M = stm(theta)

n = length(theta);
M = zeros(n);
for k=1:n/2
    M(2*k-1,2*k-1) =  theta(2*k-1);
    M(2*k  ,2*k  ) =  theta(2*k-1);
    M(2*k-1,2*k  ) =  theta(2*k);
    M(2*k  ,2*k-1) = -theta(2*k);
end

%--------------------------------------------------------------------------
function H = smm(ord)

h = [1 0];
H = repmat(h,1,ord);

%--------------------------------------------------------------------------
function Ctheta = pmm(z)

Ctheta = z';