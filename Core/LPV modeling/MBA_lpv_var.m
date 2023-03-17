function [Pyy,omega,omegaN,zeta,Psi] = MBA_lpv_var(M,xi)

%% Part 0 : Unpacking and checking input

N = size(xi,2);
n = size(M.Parameters.Theta,1);

%-- Model structure
na = M.structure.na;                                                        % AR order
pa = M.structure.pa;                                                        % Basis order

%% Part 1 : Constructing the representation basis

%-- Building the representation basis according to the basis type
switch M.structure.basis.type
    
    %-- Cosine basis
    case 'cosine'

        g = ones(pa,N);
        for j=1:pa-1
            g(j+1,:) = cos(j*2*pi*xi);
        end

    %-- Fourier basis
    case 'fourier'
        
        % The basis order must be odd when using the Fourier basis to
        % ensure numerical stability
        if mod(pa,2) == 0
            pa = pa-1;
            warning('Basis order must be odd when using the Fourier basis. Basis order set to p-1')
        else
            g = ones(pa,N);
            for j=1:(pa-1)/2
                g(2*j,:) = sin(j*pi*xi);
                g(2*j+1,:) = cos(j*pi*xi);
            end
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
        Pyy(:,:,j,i) = abs((Aden\M.InnovationsCovariance.SigmaW)/Aden);
    end 
    
    D = [ reshape(-A(:,:,:,i),n,n*na); eye(n*(na-1),n*na) ];
    [V,rho] = eig(D);
    Psi(:,:,i) = V(1:n,:);
    rho = diag(rho);
    omegaN(:,i) = abs(log(rho));
    zeta(:,i) = -cos(angle(log(rho)));
end
