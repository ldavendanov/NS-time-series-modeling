function [ym,Am,omega,error,logMarginal] = DiagonalSS3(y,M,variances)
%--------------------------------------------------------------------------
% Estimation of the modal components of a non-stationary signal using the
% Joint EKF method with instant magnitude equal to the unit
% Created by : David Avendano - April 2016
%--------------------------------------------------------------------------

n = 2*M;            % Dimension of the modal and parameter vector
N = length(y);      % Lenght of the signal

% Setting up the state space representation
System.ffun = @ffun;
System.F = @stm;
System.H = @smm;

% Setting up the noise features
R = 1;
if nargin < 3
    lambda = 1e-4;
    mu = 1e2*lambda;
else
    R = variances(1);
    lambda = variances(2);      % Variance of the parameters
    mu = variances(3);          % Variance of the mode trajectories
end
Q = lambda*eye(n+M);
Q(1:n,1:n) = mu*eye(n);

% Initializing the estimation matrices
xtt = zeros(n+M,N);
xttm = zeros(n+M,N);
Ptt = zeros(n+M,n+M,N);
Pttm = zeros(n+M,n+M,N);
K = zeros(n+M,1,N);
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
for m=1:M
    xtt(n+m) = angle(alpha(2*m-1));
end

Ptt(:,:,1) = 1e-2*eye(n+M);
Pttm(:,:,1) = Ptt(:,:,1);

%-- Kalman filter ---------------------------------------------------------
H = System.H(M);
for i=2:N
    % System matrices
    F = System.F(xtt(:,i-1)); 
    
    xttm(:,i) = System.ffun(xtt(:,i-1));
    ettm(:,i) = y(:,i) - H*xttm(:,i);
    Pttm(:,:,i) = F*Ptt(:,:,i-1)*F' + Q;

    sigma_e2 = H*Pttm(:,:,i)*H' + R;
    logMarginal(i) = -0.5*(log(sigma_e2) + ettm(:,i).^2/sigma_e2);
    
    % Kalman gain
    K(:,:,i) = Pttm(:,:,i)*H' / sigma_e2;
    
    % Correction
    xtt(:,i) = xttm(:,i) + K(:,:,i) * ettm(:,i);
    e(:,i) = y(:,i) - H*xtt(:,i);
    Ptt(:,:,i) = ( eye(n+M) - K(:,:,i)*H )*Pttm(:,:,i);

end

% Yielding resulting modal decomposition
ym = xtt(1:2:n,:);
omega = xtt(n+(1:M),:);
Am = zeros(M,N);
for m=1:M
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
function z_new = ffun(z_old)

n = length(z_old);
M = dffun_dz(z_old(2*n/3+1:n));
z_new = z_old;
z_new(1:2*n/3) = M*z_old(1:2*n/3);

%--------------------------------------------------------------------------
function F = stm(z)

n = length(z);
M = dffun_dz(z(2*n/3+1:n));
Z = dffun_dtheta(z(1:2*n/3),z(2*n/3+1:n));
F = [M Z; zeros(n/3,2*n/3) eye(n/3)];

%--------------------------------------------------------------------------
function H = smm(ord)

h = [1 0];
H = [repmat(h,1,ord) zeros(1,ord)];

%--------------------------------------------------------------------------
function M = dffun_dz(theta)

n = length(theta);
M = zeros(2*n);
for k=1:n
    M(2*k-1,2*k-1) =  cos(theta(k));
    M(2*k  ,2*k  ) =  cos(theta(k));
    M(2*k-1,2*k  ) =  sin(theta(k));
    M(2*k  ,2*k-1) = -sin(theta(k));
end

%--------------------------------------------------------------------------
function Z = dffun_dtheta(z,theta)

n = length(theta);
M = zeros(2*n);
for k=1:n
    M(2*k-1,2*k-1) = -sin(theta(k));
    M(2*k  ,2*k  ) = -sin(theta(k));
    M(2*k-1,2*k  ) =  cos(theta(k));
    M(2*k  ,2*k-1) = -cos(theta(k));
end
Zo = M*z;
Z = zeros(2*n,n);
for k=1:n
    Z(2*k-1:2*k,k) = Zo(2*k-1:2*k,1);
end