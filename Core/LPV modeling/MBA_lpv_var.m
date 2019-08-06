function [Pyy,omega,omegaN,zeta,Psi] = MBA_lpv_var(M,xi)

%% Part 0 : Unpacking and checking input

N = size(xi,2);
n = size(M.ParameterVector,1);

%-- Model structure
na = M.structure.na;                                                        % AR order
pa = M.structure.pa;                                                        % Basis order

%% Part 1 : Constructing the representation basis

%-- Building the representation basis according to the basis type
switch M.structure.basis.type
    
    %-- Fourier basis
    case 'fourier'
        
        g = ones(pa,N);
        for j=1:(pa-1)/2
            g(2*j,:) = sin(j*2*pi*xi(1,:));
            g(2*j+1,:) = cos(j*2*pi*xi(1,:));
        end
        
    %-- Hermite polynomials
    case 'hermite'
        g = ones(pa,N);
        g(2,:) = 2*xi;
        for j=3:pa
            g(j,:) = 2*xi.*g(j-1,:) - 2*(j-1)*g(j-2,:);
        end
        
end

%-- Selecting the indices of the basis to be used in the analysis
if isfield(M.structure.basis,'indices')
    g = g(M.structure.basis.indices,:);
    pa = sum(M.structure.basis.indices);
end

%% Part 2 : Calculating the frozen PSD and modal properties

Nfrec = 1024;
Pyy = zeros(n,n,Nfrec,N);
Psi = zeros(n,n*na,N);
omegaN = zeros(n*na,N);
zeta = zeros(n*na,N);

A = zeros(n,n,na,N);
for i=1:N
    A(:,:,:,i) = squeeze(g(1,i)*M.a(:,:,1,:));
    for j=2:pa
        A(:,:,:,i) = A(:,:,:,i) + g(j,i)*squeeze(M.a(:,:,j,:));
    end
end

omega = pi*(0:Nfrec-1)/Nfrec;

for i=1:N
    
    for j=1:Nfrec
        Aden = eye(n);
        for k=1:na
            Aden = Aden + A(:,:,k,i)*exp(-1i*omega(j)*k);
        end
        Pyy(:,:,j,i) = abs((Aden\M.InnovationsCovariance)/Aden);
    end 
    
    D = [ reshape(A(:,:,:,i),n,n*na); eye(n*(na-1),n*na) ];
    [V,rho] = eig(D);
    Psi(:,:,i) = V(1:n,:);
    rho = diag(rho);
    omegaN(:,i) = abs(log(rho));
    zeta(:,i) = -cos(angle(log(rho)));
end
