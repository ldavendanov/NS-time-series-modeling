function [ym,Am,omega,theta,phi] = DiagonalSS5(y,a)
%--------------------------------------------------------------------------
% Estimation of the modal components of a non-stationary signal using the
% TAR model based method 
% Created by : David Avendano - April 2016
%--------------------------------------------------------------------------

[n,N] = size(a);        % Dimension of the state space and the signal lenght
M = n/2;

% -- Initialization -------------------------------------------------------
ym = zeros(n,N);
Am = zeros(n,N);
phi = zeros(n,N);
omega = zeros(n,N);
theta = zeros(n,N);
z0 = [y(1);zeros(n-1,1)];

% -- Computation of the modal components ----------------------------------
for i=2:N
    % System matrices
    [V,theta(:,i),omega(:,i),ind] = stm(a(:,i)); 
    z0 = [y(i); z0(1:end-1)];
    ym(:,i) = V\z0;
    ym(:,i) = ym(ind,i);
    Am(:,i) = abs(ym(:,i));
    phi(:,i) = angle(ym(:,i));
    
    for m=1:2*M
        e = phi(m,i)-phi(m,i-1);
        if e < 0, 
            ym(m,i) = -ym(m,i); 
            phi(m,i) = abs(e) + phi(m,i-1);
        end
        
    end
end

% Yielding resulting modal decomposition
ym = real(ym(M+1:end,:));
Am = Am(M+1:end,:);
omega = omega(M+1:end,:);

%--------------------------------------------------------------------------
function [V,theta,omega,ind] = stm(a)

% Eigenvalue decomposition of the companion form
A = compan([1; a]);                         % Companion form matrix
[V,D] = eig(A);                             % Eigenvalues and eigenvectors
theta = diag(D);                            % Extracting the eigenvalues
omega = angle(theta);
[omega,ind] = sort(omega);
theta = theta(ind);                         % Sorted eigenvalues
