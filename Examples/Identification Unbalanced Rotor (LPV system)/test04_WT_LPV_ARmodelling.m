clear
close all
clc

% Physical properties of the system
System.mb = 60*[1.00 1 1];                                                  % Blade mass (kg)
System.kb = 4e4*[1.05 1 1];                                                    % Blade torsional stiffness (N/rad)
System.cb = 2e-1*[1 1 1];                                                   % Blade torsional damping (N.s/rad)
System.lb = 2;                                                              % Blade length (m)

System.Mt = 4e3;                                                            % Tower mass (kg)
System.Jo = 9e2;                                                            % Drivetrain mass moment of inertia (kg.m^2)
System.Kt = [6e4 8e4];                                                      % Tower stiffness [horizontal vertical] (N/m)
System.Ct = [5 5];                                                          % Tower damping coeff [horizontal vertical] (N.s/m)

System.Cpi = [1e2 1e3];                                                     % Constants of PI control of rotor speed [Cp Ci]

OmegaRef = 2*pi*10/60;                                                      % Rotational speed (rad/s)

% Simulation properties
fs0 = 40;                                                                   % Sampling rate (Hz)
T = 60*60;                                                                  % Sampling period (s)
N = (T*fs0)+1;                                                              % Number of samples
t = (0:N-1)/fs0;                                                            % Time vector

% Excitation properties
[b,a] = butter(4,[0.01 0.1]);                                                     % Parameters of beamforming filter
TauMn = 2;                                                                 % Mean value of excitation (N.m)
TauSD = 0.02;                                                                % Standard deviation of excitation (N.m)
TauM = TauMn + TauSD*filter(b,a,randn(N,3))';                               % Excitation time series - Torque at each blade (N.m)

% Integrate EOM
[t,z] = ode45( @(t,z)WTRotorLinEOM( t,z, System, OmegaRef, TauM, fs0 ), t, zeros(10,1) );

% Calculate acceleration response
y = zeros(N,5);
for i=1:N
    dz = WTRotorLinEOM( i/fs0, z(i,:)', System, OmegaRef, TauM, fs0 );
    y(i,:) = dz(6:10);
end


%% Downsample acceleration response
close all
clc

fs = 10;                                                                    % Analysis sampling frequency (Hz)
ys = resample(y,fs,fs0);

beta = mod(OmegaRef*t,2*pi)/(2*pi);
beta = beta(1:4:end);

%% Estimate LPV-AR model
close all
clc

addpath(genpath('..\..\Core'))

T = 2*pi/OmegaRef;
ind_train = 2e3+(1:100*T*fs);
signals.response = ys(ind_train,1)';                                        % Response ( horizontal tower acceleration )
signals.scheduling_variables = beta(ind_train)';                            % Scheduling variable ( rotor azimuth )

na_max = 30;
pa_max = 25;

na = 1:na_max;
pa = 1:2:pa_max;

rss_sss = zeros(numel(na),numel(pa));
bic = zeros(numel(na),numel(pa));

fN = zeros(na_max*pa_max,numel(na),numel(pa));
zeta = zeros(na_max*pa_max,numel(na),numel(pa));

for i = 1:numel(na)
    for j=1:numel(pa)

        structure.na = na(i);
        structure.pa = pa(j);

        options.basis.type = 'fourier';                                             % Type of functional basis
        options.basis.ind_ba = 1:structure.pa;

        M = estimate_lpv_ar(signals,structure,options);
        rss_sss(i,j) = M.Performance.rss_sss;
        bic(i,j) = M.Performance.bic;

    end
end

%% Analysis of model order selection results
close all
clc

clr = parula(numel(pa));

na_opt = 16;
pa_opt = 9;

figure('Position',[100 100 900 600])
tiledlayout(2,2)

nexttile
for i=1:numel(pa)
    semilogy(na,rss_sss(:,i)*100,'Color',clr(i,:),'LineWidth',1.5)
    hold on
end
xlabel('Model order')
ylabel('RSS/SSS (%)')
grid on
xlim([5 inf])

nexttile
for i=1:numel(pa)
    plot(na,bic(:,i),'Color',clr(i,:),'LineWidth',1.5)
    hold on
end
xlabel('Model order')
ylabel('BIC')
grid on
xlim([5 inf])

nexttile
semilogy(pa,(rss_sss(na_opt,:)'*100),'-o')
xlabel('Basis order')
ylabel('RSS/SSS (%)')
grid on

nexttile
plot(pa,bic(na_opt,:),'-o')
xlabel('Basis order')
ylabel('BIC')
grid on

%% Model based analysis
close all
clc

na_opt = 16;
pa_opt = 9;

structure.na = na_opt;
structure.pa = pa_opt;

options.basis.type = 'fourier';                                             % Type of functional basis
options.basis.ind_ba = 1:structure.pa;

M = estimate_lpv_ar(signals,structure,options);

m = 200;
Xi_range = linspace(0,1,m);
[Pyy,omega,omegaN,zeta] = MBA_lpv_ar(M,Xi_range);

figure
surf(Xi_range,omega*fs/(2*pi),10*log10(Pyy),'LineStyle','none')
axis ij

figure
tiledlayout(3,1)

nexttile
pwelch(signals.response,hann(2^10),2^10*3/4,2^10,fs)

nexttile(2,[2 1])
plot(omegaN'*fs/(2*pi),zeta','.b')
xlim([0 fs/2])
grid on
% ylim(0.2*[-1 1])