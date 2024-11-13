function dz = UnbalancedRotorODE4dof(t,z,OmegaRef,Mo,fs)

% System settings
m = [1000 100];               % Mass (kg)
k = [80000 40000 1 0000];       % Stiffness (N/m)
c = [0 0 0.00];    % Damping (N*s/m)
l2 = 0.2;                   % Rotor radius before deformation (m)

% System variables
Omega = z(8);               % Rotor speed (rad/s)
theta = z(4);               % Rotor angle (rad)
d2 = l2 + z(3);             % Deformed rotor radius (m)
Fc = m(2)*d2*Omega^2;       % Centrifugal force of rotor (N)

% Rotor speed control
Cp = 4e1;                   % Proportional coefficient
Ci = 2e2;                   % Integral coefficient
Tau = z(9);                 % Applied torque
err = OmegaRef - Omega;

ind = max(1,round(t*fs));

% System matrices
M = [             sum(m)                  0  m(2)*cos(theta) -m(2)*d2*sin(theta);
                       0             sum(m)  m(2)*sin(theta)  m(2)*d2*cos(theta);
         m(2)*cos(theta)    m(2)*sin(theta)             m(2)                   0;
     -m(2)*d2*sin(theta) m(2)*d2*cos(theta)                0           m(2)*d2^2];

f = [ Fc*cos(theta) + 2*m(2)*sin(theta)*z(7)*z(8) - k(1)*z(1) - c(1)*z(5); 
      Fc*sin(theta) - 2*m(2)*cos(theta)*z(7)*z(8) - k(2)*z(2) - c(2)*z(6); 
      Fc - k(3)*z(3) - c(3)*z(7);
      Mo(ind) + Tau - 2*m(2)*z(7)*z(8)*d2];

dx = M\f;
dz = [z(5:8); dx; Ci*err - Cp*dx(4)];

end