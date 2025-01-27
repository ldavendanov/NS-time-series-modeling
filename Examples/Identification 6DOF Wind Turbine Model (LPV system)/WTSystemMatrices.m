function [M,C,K] = WTSystemMatrices( System, Omega, t )

% Blade angle and sine/cosine matrices
Psi = Omega*t + [0 2*pi/3 -2*pi/3];
cp = cos(Psi);
sp = sin(Psi);

% Mass matrix
Mtb = System.mb.*System.lb.*[-sp; cp];
Mt = sum(System.mb)+System.Mt*eye(2);
Mb = diag( System.mb.*System.lb.^2 );

M = [  Mt  Mtb;
      Mtb'  Mb];

% Damping matrix
Ctb = -2*Omega*System.mb.*System.lb.*[cp; sp];
Ct = diag(System.Ct);
Cb = diag(System.cb);

C = [         Ct Ctb;
       zeros(3,2) Cb];

% Stiffness matrix
Ktb = Omega^2*System.mb.*System.lb.*[sp; -cp];
Kt = diag(System.Kt);
Kb = diag(System.kb);

K = [         Kt Ktb;
       zeros(3,2) Kb];

end