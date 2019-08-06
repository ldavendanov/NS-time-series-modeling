function [ym,Am,omega,theta,error,logMarginal] = DiagonalSS1(y,M,variances)
%--------------------------------------------------------------------------
% Estimation of the modal components of a non-stationary signal using the
% Joint EKF method 
% Created by : David Avendano - April 2016
%--------------------------------------------------------------------------

n = 2*M;            % Dimension of the modal and parameter vector
N = length(y);      % Lenght of the signal

% Setting up the state space representation
System.ffun = @ffun;
System.F = @stm;
System.H = @smm;

% Setting up the noise features
if nargin < 3
    R = 1;
    lambda = 1e-4;
    mu = 5e2*lambda;
else
    R = variances(1);
    lambda = variances(2);      % Variance of the parameters
    mu = variances(3);          % Variance of the mode trajectories
end
Q = lambda*eye(2*n);
Q(1:n,1:n) = mu*eye(n);

% Initializing the estimation matrices
xtt = zeros(2*n,N);
xttm = zeros(2*n,N);
Ptt = zeros(2*n,2*n,N);
Pttm = zeros(2*n,2*n,N);
K = zeros(2*n,1,N);
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
    xtt(2*M+2*m-1) = real(alpha(2*m-1));
    xtt(2*M+2*m) = imag(alpha(2*m-1));
end

Ptt(:,:,1) = 1e-2*eye(2*n);
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
    xtt(:,i) = xttm(:,i) + K(:,:,i) * ( y(:,i) - H*xttm(:,i)  );
    e(:,i) = y(:,i) - H*xtt(:,i);
    Ptt(:,:,i) = ( eye(2*n) - K(:,:,i)*H )*Pttm(:,:,i);

end

% Yielding resulting modal decomposition
ym = xtt(1:2:2*M,:);
theta = xtt(2*M+1:4*M,:);
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
function z_new = ffun(z_old)

n = length(z_old);
M = dffun_dz(z_old(n/2+1:n));
z_new = z_old;
z_new(1:n/2) = M*z_old(1:n/2);

%--------------------------------------------------------------------------
function F = stm(z)

n = length(z);
M = dffun_dz(z(n/2+1:n));
Z = dffun_dtheta(z(1:n/2));
F = [M Z; zeros(n/2) eye(n/2)];

%--------------------------------------------------------------------------
function H = smm(ord)

h = [1 0];
H = [repmat(h,1,ord) zeros(1,2*ord)];

%--------------------------------------------------------------------------
function M = dffun_dz(theta)

n = length(theta);
M = zeros(n);
for k=1:n/2
    M(2*k-1,2*k-1) =  theta(2*k-1);
    M(2*k  ,2*k  ) =  theta(2*k-1);
    M(2*k-1,2*k  ) =  theta(2*k);
    M(2*k  ,2*k-1) = -theta(2*k);
end

%--------------------------------------------------------------------------
function Z = dffun_dtheta(z)

n = length(z);
Z = zeros(n);
for k=1:n/2
    Z(2*k-1,2*k-1) =  z(2*k-1);
    Z(2*k  ,2*k  ) = -z(2*k-1);
    Z(2*k-1,2*k  ) =  z(2*k);
    Z(2*k  ,2*k-1) =  z(2*k);
end