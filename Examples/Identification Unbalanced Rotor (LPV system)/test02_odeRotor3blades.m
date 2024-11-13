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

OmegaRef = 2*pi*12/60;                                                      % Rotational speed (rad/s)

% Simulation properties
fs0 = 40;                                                                   % Sampling rate (Hz)
T = 20*60;                                                                  % Sampling period (s)
N = (T*fs0)+1;                                                              % Number of samples
t = (0:N-1)/fs0;                                                            % Time vector

% Excitation properties
[b,a] = butter(4,[0.01 0.05]);                                                     % Parameters of beamforming filter
TauMn = 10;                                                                 % Mean value of excitation (N.m)
TauSD = 0.4;                                                                % Standard deviation of excitation (N.m)
TauM = TauMn + TauSD*filter(b,a,randn(N,3))';                               % Excitation time series - Torque at each blade (N.m)

% Integrate EOM
[t,z] = ode45( @(t,z)WTRotorEOM( t,z, System, OmegaRef, TauM, fs0 ), t, zeros(13,1) );

% Calculate acceleration response
y = zeros(N,5);
for i=1:N
    dz = WTRotorEOM( i/fs0, z(i,:)', System, OmegaRef, TauM, fs0 );
    y(i,:) = dz(7:11);
end

beta = mod(z(:,6),2*pi)/(2*pi);

%% Downsampling
close all
clc

fs = 10;
y = resample(y,fs,fs0);
t = t(1:fs0/fs:end);
beta = beta(1:fs0/fs:end);
N = length(t);

%% Remove 1P component
close all
clc

S = [cos(2*pi*beta) sin(2*pi*beta)];
Yp = 2*S'*y/N;
yf = y-S*Yp;

figure
plot(t,S*Yp,t,y(:,1))

figure
pwelch([y(:,1) y(:,1)-S*Yp])



%%
close all
clc

figure('Position',[100 100 600 600])
tiledlayout(5,1)

for i=1:5
    nexttile
    plot(t/60,y(:,i))
    grid on
    xlim([10 20])
end

% figure('Position',[700 100 600 600])
% tiledlayout(2,1)

% nexttile
% plot(t/60,z(:,12)*60/(2*pi))
% hold on
% plot([0 T]/60, OmegaRef*[1 1]*60/(2*pi), 'k')
% xlim([10 20])
% grid on
% 
% nexttile
% plot(t/60,z(:,13))
% hold on
% plot(t/60,TauM)
% xlim([10 20])
% grid on

%% Frequency analysis
clc
close all

Nf = 2^12;
figure
pwelch(y(4001:end,:),hann(Nf),3*Nf/4,Nf,fs)


figure
tiledlayout(3,2)

for i=1:5
    Nf = 2^11;
    [Syy,ff,tt] = spectrogram( y(:,i), gausswin(Nf,4), Nf-1, Nf, fs );

    nexttile
    imagesc(tt/60,ff,log10(abs(Syy)))
    axis xy

end

%% Estimate LPV-AR model
close all
clc

addpath(genpath('..\..\Core'))

ind_train = 2e3+(1:1e4);
signals.response = y(ind_train,3)';                                          % Response ( edgewise acceleration )
signals.scheduling_variables = beta(ind_train)';                            % Scheduling variable ( rotor azimuth )

na_max = 20;
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

        % A_lft = [M.Parameters.Theta;
        %           eye((na(i)-1)*pa(j),na(i)*pa(j))];
        % D = eig(A_lft);
        % 
        % fN(1:na(i)*pa(j),i,j) = abs(log(D))*fs/(2*pi);
        % zeta(1:na(i)*pa(j),i,j) = -cos(angle(log(D)));

    end
end

%% Analysis of model order selection results
close all
clc

na_opt = 14;
pa_opt = 9;

figure('Position',[100 100 900 600])
tiledlayout(2,2)

nexttile
semilogy(na,(rss_sss'*100))
xlabel('Model order')
ylabel('RSS/SSS (%)')
grid on

nexttile
semilogy(pa,(rss_sss(na_opt,:)'*100),'-o')
xlabel('Basis order')
ylabel('RSS/SSS (%)')
grid on

nexttile
plot(na,bic')
xlabel('Model order')
ylabel('BIC')
grid on

nexttile
plot(pa,bic(na_opt,:),'-o')
xlabel('Basis order')
ylabel('BIC')
grid on

%% Frequency stabilization diagrams
close all
clc

zeta_max = 0.2;
z_lvl = linspace(0,zeta_max);
clr = parula(numel(z_lvl)-1);

figure('Position',[100 100 600 600])
tiledlayout(3,1)

Nf = 2^12;
nexttile
pwelch(y,hann(Nf),3*Nf/4,Nf,fs)

nexttile(2,[2 1])
FN = squeeze(fN(:,:,(pa_opt-1)/2));  FN = FN(:);
Z = squeeze(zeta(:,:,(pa_opt-1)/2));  Z = Z(:);
NA = squeeze( repmat(na, na_max*pa_max, 1) ); NA = NA(:);

for i=1:numel(z_lvl)-1
    ind = Z > z_lvl(i) & Z <= z_lvl(i+1);
    plot(FN(ind),NA(ind),'.','Color',clr(i,:))
    hold on
end
grid on
xlim([0 fs/2]), ylim([0 na_max])
xlabel('Frequency (Hz)')
ylabel('AR order')
cbar = colorbar;
cbar.Ticks = (0:5)/5;
cbar.TickLabels = linspace(0,zeta_max*100,6);
cbar.Label.String = 'Damping ratio (%)';


figure('Position',[800 100 600 600])
tiledlayout(3,1)

Nf = 2^12;
nexttile
pwelch(y,hann(Nf),3*Nf/4,Nf,fs)

nexttile(2,[2 1])
FN = squeeze(fN(:,na_opt,:));  FN = FN(:);
Z = squeeze(zeta(:,na_opt,:));  Z = Z(:);
PA = squeeze( repmat(pa, na_max*pa_max, 1) ); PA = PA(:);

for i=1:numel(z_lvl)-1
    ind = Z > z_lvl(i) & Z <= z_lvl(i+1);
    plot(FN(ind),PA(ind),'.','Color',clr(i,:))
    hold on
end
grid on
xlim([0 fs/2]), ylim([0 pa_max])
ylabel('Basis order')
cbar = colorbar;
cbar.Ticks = (0:5)/5;
cbar.TickLabels = linspace(0,zeta_max*100,6);
cbar.Label.String = 'Damping ratio (%)';

%% Model based analysis
close all
clc

na_opt = 24;
pa_opt = 7;

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
ylim(0.2*[-1 1])