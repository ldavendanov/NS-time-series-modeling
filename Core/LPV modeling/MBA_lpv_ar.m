function [Pyy,omega,omegaN,zeta] = MBA_lpv_ar(M,xi)

%% Part 0 : Unpacking and checking input

N = size(xi,2);

%-- Model structure
na = M.Structure.na;                                                        % AR order
pa = M.Structure.pa;                                                        % Basis order
ps = M.Structure.ps;                                                        % Basis order

%% Part 1 : Constructing the representation basis

%-- Representation basis for the AR parameters
if isfield(M.Structure.basis,'ind_ba')
    ba = M.Structure.basis.ind_ba;
    Gba = lpv_basis(xi,ba,M.Structure.basis);
else
    ba = 1:pa;
    Gba = lpv_basis(xi,ba,M.Structure.basis);
end

%-- Representation basis for the innovations variance
if ps > 1
    if isfield(M.Structure.basis,'ind_bs')
        bs = M.Structure.basis.ind_bs;
        Gbs = lpv_basis(xi,bs,M.Structure.basis);
    else
        bs = 1:ps;
        Gbs = lpv_basis(xi,bs,M.Structure.basis);
    end
end

%% Part 2 : Calculating the frozen PSD and modal properties

Nfrec = 1024;
Pyy = zeros(Nfrec,N);
omegaN = zeros(na,N);
zeta = zeros(na,N);

A = M.ar_part.a'*Gba(M.Structure.basis.ind_ba,:);
omega = 2*pi*(0:Nfrec-1)/(2*Nfrec);

if M.Structure.ps > 1
    sigmaW2 = M.InnovationsVariance.S.Parameters.Theta*Gbs(M.Structure.basis.ind_bs,:);
else
    sigmaW2 = M.InnovationsVariance.sigmaW2*ones(1,N);
end

for i=1:N
    den = [1; A(:,i)];
    Pyy(:,i) = sigmaW2(i)*abs( freqz(1,den, Nfrec ) ).^2;
    rho = roots(den);
    omegaN(:,i) = abs(log(rho));
    zeta(:,i) = -cos(angle(log(rho)));
end
