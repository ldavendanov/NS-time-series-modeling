function f = WTLoadVector( System, Omega, t, TauM, fs )

ind = max(round(t*fs),1);

Psi = Omega*t + [0 2*pi/3 -2*pi/3];
cp = cos(Psi)';
sp = sin(Psi)';

f = [ Omega^2* (System.mb.*System.lb)*cp
      Omega^2* (System.mb.*System.lb)*sp
      TauM(:,ind)];

end