clear
close all
clc

% Define WT rotor properties
System = WTrotorProperties;                                                 % Physical properties of the WT
n = 5;

% Simulation properties
fs0 = 40;                                                                   % Sampling rate (Hz)
T = 10*60;                                                                  % Sampling period (s)
N = (T*fs0)+1;                                                              % Number of samples
t = (0:N-1)/fs0;                                                            % Time vector

Omega = 2*pi*linspace(0,20)/60;                                             % Vector with rotor speeds (0 to 20 RPM)

% Create excitation
TauM = SimulateExcitation(N,fs0);                                           % Excitation time series - Torque at each blade (N.m)

% Frequency analysis parameters
Nf = 2^12;                                                                  % PSD Blocksize
Pyy = zeros(Nf/2+1,n,numel(Omega));                                         % Initialized PSD array

% Calculate response PSD for multiple rotational speeds
parfor k=1:numel(Omega)

    % Integrate EOM
    [~,z] = ode45( @(t,z)WTRotorLinEOM( t,z, System, Omega(k), TauM, fs0 ), t, zeros(10,1) );

    % Calculate acceleration response
    y = zeros(N,n);
    for i=1:N
        dz = WTRotorLinEOM( i/fs0, z(i,:)', System, Omega(k), TauM, fs0 );
        y(i,:) = dz(6:10);
    end

    Pyy(:,:,k) = pwelch( y, hann(Nf), Nf*3/4, Nf, fs0 );

end

%% Plot Campbell diagram
close all
clc

freq = fs0*(0:Nf/2)/Nf;

figure
tiledlayout(1,n)

for i=1:n
    nexttile
    imagesc(60*Omega/(2*pi),freq, log10( squeeze(Pyy(:,i,:)) ))
    axis xy
    ylim([0 4])
end

