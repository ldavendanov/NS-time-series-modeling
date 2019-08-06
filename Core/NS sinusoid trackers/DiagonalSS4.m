function [ym,Am,omega,theta,error] = DiagonalSS4(y,a,variance)
%--------------------------------------------------------------------------
% Estimation of the modal components of a non-stationary signal using the
% TAR model based method 
% Created by : David Avendano - April 2016
%--------------------------------------------------------------------------

[n,N] = size(a);        % Dimension of the state space and the signal lenght
M = n/2;

% Setting up the state space representation
System.F = @stm;
System.H = @smm;

% Setting up the noise features
R = 1;
if nargin < 3
    lambda = 100;
else
    lambda = variance;
end
Q = lambda*eye(n);

% Initializing the estimation matrices
xtt = zeros(n,N);
xttm = zeros(n,N);
Ptt = zeros(n,n,N);
Pttm = zeros(n,n,N);
K = zeros(n,1,N);
e = zeros(1,N);
ettm = zeros(1,N);
theta = zeros(2*M,N);
omega = zeros(M,N);

% Initialization
A = compan([1; a(:,1)]);
[V,~] = eig(A);
Wo = [1 1; 1i -1i];
W = kron(eye(M),Wo);
z0 = y(n:-1:1);
z0 = z0(:);

% Setting up the initial modal values
xtt(:,1) = real((W/V)*z0);

Ptt(:,:,1) = 1e4*eye(n);
Pttm(:,:,1) = Ptt(:,:,1);

%-- Kalman filter ---------------------------------------------------------
H = System.H(M);
for i=2:N
    % System matrices
    [F,theta(:,i),omega(:,i)] = System.F(a(:,i)); 
    
    xttm(:,i) = F*xtt(:,i-1);
    ettm(:,i) = y(:,i) - H*xttm(:,i);
    Pttm(:,:,i) = F*Ptt(:,:,i-1)*F' + Q;
    
    % Kalman gain
    K(:,:,i) = Pttm(:,:,i)*H' / ( H*Pttm(:,:,i)*H' + R );
    
    % Correction
    xtt(:,i) = xttm(:,i) + K(:,:,i) * ettm(:,i);
    e(:,i) = y(:,i) - H*xtt(:,i);
    Ptt(:,:,i) = ( eye(n) - K(:,:,i)*H )*Pttm(:,:,i);

end

% Yielding resulting modal decomposition
ym = xtt(1:2:2*M,:);
Am = zeros(M,N);
for m=1:M
    Am(m,:) = sqrt( xtt(2*m-1,:).^2 + xtt(2*m,:).^2 );
end
error.prior = ettm;
error.posterior = e;

%--------------------------------------------------------------------------
function [F,theta,omega] = stm(a)

M = length(a)/2;
A = compan([1; a]);
theta = eig(A);
[omega,ind] = sort(abs(angle(theta)));
omega = omega(1:2:end);

Wo = [1 1; 1i -1i];
W = kron(eye(M),Wo);
F = real(W*diag(theta(ind))/W);

%--------------------------------------------------------------------------
function H = smm(ord)

h = [1 0];
H = repmat(h,1,ord);