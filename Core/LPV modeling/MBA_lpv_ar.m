function [Pyy,omega,omegaN,zeta] = MBA_lpv_ar(M,xi)

%% Part 0 : Unpacking and checking input

N = size(xi,2);

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
            g(2*j,:) = sin(j*xi(1,:));
            g(2*j+1,:) = cos(j*xi(1,:));
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
Pyy = zeros(Nfrec,N);
omegaN = zeros(na,N);
zeta = zeros(na,N);

A = M.a'*g;
omega = 2*pi*(0:Nfrec-1)/(2*Nfrec);


for i=1:N
    den = [1; A(:,i)];
    Pyy(:,i) = M.InnovationsVariance*abs( freqz(1,den, Nfrec ) ).^2;
    rho = roots(den);
    omegaN(:,i) = abs(log(rho));
    zeta(:,i) = -cos(angle(log(rho)));
end
