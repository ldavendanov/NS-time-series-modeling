function System = WTrotorProperties

% Physical properties of the system
System.mb = 60*[1.00 1 1];                                                  % Blade mass (kg)
System.kb = 4e4*[1.05 1 1];                                                 % Blade torsional stiffness (N/rad)
System.cb = 2e-1*[1 1 1];                                                   % Blade torsional damping (N.s/rad)
System.lb = 2;                                                              % Blade length (m)

System.Mt = 4e3;                                                            % Tower mass (kg)
System.Jo = 9e2;                                                            % Drivetrain mass moment of inertia (kg.m^2)
System.Kt = [6e4 8e4];                                                      % Tower stiffness [horizontal vertical] (N/m)
System.Ct = [5 5];                                                          % Tower damping coeff [horizontal vertical] (N.s/m)

System.Cpi = [1e2 1e3];                                                     % Constants of PI control of rotor speed [Cp Ci]

end