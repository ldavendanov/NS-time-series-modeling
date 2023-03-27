function [Pyy,omega,omegaN,zeta,Psi] = MBA_lpv_var(M,xi)

%% Part 0 : Unpacking and checking input

N = size(xi,2);
n = size(M.Parameters.Theta,1);

%-- Model structure
na = M.structure.na;                                                        % AR order
pa = M.structure.pa;                                                        % Basis order

g = lpv_basis(xi,1:pa,M.structure.basis);

%% Part 1 : Calculating the frozen PSD and modal properties

Nfrec = 1024;
Pyy = zeros(n,n,Nfrec,N);
Psi = zeros(n,n*na,N);
omegaN = zeros(n*na,N);
zeta = zeros(n*na,N);

A = zeros(n,n,na,N);
for i=1:N
    A(:,:,:,i) = squeeze(g(1,i)*M.a.projection(:,:,1,:));
    for j=2:pa
        A(:,:,:,i) = A(:,:,:,i) + g(j,i)*squeeze(M.a.projection(:,:,j,:));
    end
end

omega = pi*(0:Nfrec-1)/Nfrec;

for i=1:N
    
    for j=1:Nfrec
        Aden = eye(n);
        for k=1:na
            Aden = Aden + A(:,:,k,i)*exp(-1i*omega(j)*k);
        end
        Pyy(:,:,j,i) = abs((Aden\M.InnovationsCovariance.SigmaW)/Aden);
    end 
    
    D = [ reshape(-A(:,:,:,i),n,n*na); eye(n*(na-1),n*na) ];
    [V,rho] = eig(D);
    Psi(:,:,i) = V(1:n,:);
    rho = diag(rho);
    omegaN(:,i) = abs(log(rho));
    zeta(:,i) = -cos(angle(log(rho)));
end
