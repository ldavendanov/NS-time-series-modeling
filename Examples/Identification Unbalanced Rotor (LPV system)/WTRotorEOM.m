%% ------------------------------------------------------------------------
function dz = WTRotorEOM( t, z, System, OmegaRef, TauM, fs )

m = [System.mb System.Mt];
Jo = System.Jo;
k = [System.kb System.Kt];
c = [System.cb System.Ct];
C = System.Cpi;
l = System.lb;

ind = max(1,round(t*fs));
q = z(1:6);
dq = z(7:12);
Mo = z(13);
alpha = q(3:5)+q(6)+[0 2*pi/3 -2*pi/3]';
alpha_dot = dq(3:5)+dq(6);

f_int = [-k(4)*q(1) - c(4)*dq(1) + l*m(1:3)*( alpha_dot.^2.*cos(alpha) )
         -k(5)*q(2) - c(5)*dq(2) + l*m(1:3)*( alpha_dot.^2.*sin(alpha) )
         -k(1)*q(3) - c(1)*dq(3)
         -k(2)*q(4) - c(2)*dq(4)
         -k(3)*q(5) - c(3)*dq(5)
          Mo];
f_ext = [0 
         0 
         TauM(:,ind) 
         0];
M = MassMatrix( alpha, m, l, Jo );


dz = [ dq;
       M\(f_int+f_ext)
       0];
dz(end) = C(1)*(OmegaRef-dq(6)) - C(2)*dz(12);

end

function M = MassMatrix( alpha, m, l, Jo )

M = [               sum(m)                     0 -m(1)*l*sin(alpha(1)) -m(2)*l*sin(alpha(2)) -m(3)*l*sin(alpha(3))  -l*m(1:3)*sin(alpha)
                         0                sum(m)  m(1)*l*cos(alpha(1))  m(2)*l*cos(alpha(2))  m(3)*l*cos(alpha(3))   l*m(1:3)*cos(alpha)
     -m(1)*l*sin(alpha(1))  m(1)*l*cos(alpha(1))              m(1)*l^2                     0                     0              m(1)*l^2
     -m(2)*l*sin(alpha(2))  m(2)*l*cos(alpha(2))                     0              m(2)*l^2                     0              m(2)*l^2
     -m(3)*l*sin(alpha(3))  m(3)*l*cos(alpha(3))                     0                     0              m(3)*l^2              m(3)*l^2
      -l*m(1:3)*sin(alpha)   l*m(1:3)*cos(alpha)              m(1)*l^2              m(2)*l^2              m(3)*l^2  Jo + sum(m(1:3))*l^2];

end