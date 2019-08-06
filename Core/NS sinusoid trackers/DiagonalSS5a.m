function [ym,Am,omega,theta,phi] = DiagonalSS5a(y,a)
%--------------------------------------------------------------------------
% Estimation of the modal components of a non-stationary signal using the
% TAR model based method 
% Created by : David Avendano - April 2016
%--------------------------------------------------------------------------

[n,N] = size(a);        % Dimension of the state space and the signal lenght
M = n/2;

% -- Initialization -------------------------------------------------------
ym = zeros(M,N);
Am = zeros(M,N);
omega = zeros(M,N);
theta = zeros(n,N);
z0 = [y(1); zeros(n-1,1)];
phi = zeros(M,N);

% Rotation matrix
c = [1 zeros(1,2*M-1)];

% -- Computation of the modal components ----------------------------------
for i=2:N
    % Regression vector
    z0 = [y(i); z0(1:n-1)];
    
    % System matrices
    A = compan([1;a(:,i)]);
    [V,D] = eig(A);
    H = diag(V'*c')/V;
    y0 = H*z0;
    phi(:,i) = angle(y0(1:2:end));
    
    % Extracting eigenvalues and instantaneous frequencies
    theta(:,i) = diag(D);
    omega(:,i) = abs(angle(theta(1:2:end,i)));
    Am(:,i) = 2*abs(y0(1:2:end));
    
    % Sorting the frequencies, eigenvalues and modes
    [omega(:,i),ind] = sort(omega(:,i));
    Am(:,i) = Am(ind,i);
    ym(:,i) = real(y0(1:2:end) + y0(2:2:end));
    ym(:,i) = ym(ind,i);
end
