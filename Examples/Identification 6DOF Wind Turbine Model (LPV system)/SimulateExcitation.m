function TauM = SimulateExcitation(N,fs0)

% Excitation properties
[b,a] = butter(4,[0.4 4]/fs0);                                              % Parameters of beamforming filter
TauMn = 2;                                                                  % Mean value of excitation (N.m)
TauSD = 0.02;                                                               % Standard deviation of excitation (N.m)
TauM = TauMn + TauSD*filter(b,a,randn(N,3))';                               % Excitation time series - Torque at each blade (N.m)

end