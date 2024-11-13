function dz = UnbalancedRotorODE(~,z,OmegaRef)

% System settings
m = [10 0.2];               % Mass (kg)
k = [1000 200];             % Stiffness (N/m)
c = 0.01*[1 0.1];                % Damping (N*s/m)
l2 = 0.2;                   % Rotor radius before deformation (m)

% System variables
Omega = z(6);               % Rotor speed (rad/s)
theta = z(3);               % Rotor angle (rad)
d2 = l2 + z(2);             % Deformed rotor radius (m)
Fc = m(2)*d2*Omega^2;       % Centrifugal force of rotor (N)

% Rotor speed control
Cp = 4e-1;                   % Proportional coefficient
Ci = 2e0;                   % Integral coefficient
Tau = z(7);                 % Applied torque
err = OmegaRef - Omega;

% System matrices
M = [            sum(m) m(2)*sin(theta) m(2)*d2*cos(theta);
        m(2)*sin(theta)            m(2)                  0;
     m(2)*d2*cos(theta)               0          m(2)*d2^2];

f = [ Fc*sin(theta) - 2*m(2)*cos(theta)*z(5)*z(6) - k(1)*z(1) - c(1)*z(4); 
      Fc - k(2)*z(2) - c(2)*z(5);
      Tau - 2*m(2)*z(5)*z(6)*d2];

dx = M\f;
dz = [z(4:6); dx; Ci*err - Cp*dx(3)];

end