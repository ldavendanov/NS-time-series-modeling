function [Pyy,H,omega,omegaN,zeta,Psi] = MBA_lpv_varx(M,xi)

%% Part 0 : Unpacking and checking input

N = size(xi,2);
n = size(M.a,1);
m = size(M.b,2);

%-- Model structure
na = M.structure.na;                                                        % AR order
pa = M.structure.pa;                                                        % Basis order

g = lpv_basis(xi,1:pa,M.structure.basis);

%% Part 1 : Calculating the frozen PSD and modal properties

Nfrec = 1024;
Pyy = zeros(n,n,Nfrec,N);
H = zeros(n,m,Nfrec,N);
Psi = zeros(n,n*na,N);
omegaN = zeros(n*na,N);
zeta = zeros(n*na,N);

A = zeros(n,n,na,N);
B = zeros(n,m,na,N);
for i=1:N
    A(:,:,:,i) = squeeze(g(1,i)*M.a(:,:,1,:));
    B(:,:,:,i) = squeeze(g(1,i)*M.b(:,:,1,:));
    for j=2:pa
        D1 = g(j,i)*(M.a(:,:,j,:));
        D2 = g(j,i)*(M.b(:,:,j,:));
        for k=1:na
            A(:,:,k,i) = A(:,:,k,i) + D1(:,:,1,k);
            B(:,:,k,i) = B(:,:,k,i) + D2(:,:,1,k);
        end
    end
end

omega = pi*(0:Nfrec-1)/Nfrec;

for i=1:N
    
    for j=1:Nfrec
        Aden = eye(n);
        Bnum = zeros(n,m);
        for k=1:na
            Aden = Aden + A(:,:,k,i)*exp(-1i*omega(j)*k);
            Bnum = Bnum + B(:,:,k,i)*exp(-1i*omega(j)*k);
        end
        Pyy(:,:,j,i) = abs((Aden\M.InnovationsCovariance.SigmaW)/Aden);
        H(:,:,j,i) = Aden\Bnum;
    end 
    
    D = [ reshape(-A(:,:,:,i),n,n*na); eye(n*(na-1),n*na) ];
    [V,rho] = eig(D);
    Psi(:,:,i) = V(1:n,:);
    rho = diag(rho);
    omegaN(:,i) = abs(log(rho));
    zeta(:,i) = -cos(angle(log(rho)));

    [omegaN(:,i),ind] = sort(omegaN(:,i));
    zeta(:,i) = zeta(ind,i);
end
