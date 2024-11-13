clear
close all
clc

% Define WT rotor properties
System = WTrotorProperties;                                                 % Physical properties of the WT
n = 5;

OmegaRef = 2*pi*10/60;                                                      % Rotational speed (rad/s)

% Simulation properties
fs0 = 40;                                                                   % Sampling rate (Hz)
T = 20*60;                                                                  % Sampling period (s)
N = (T*fs0)+1;                                                              % Number of samples
t = (0:N-1)/fs0;                                                            % Time vector

% Create excitation
TauM = SimulateExcitation(N,fs0);                                           % Excitation time series - Torque at each blade (N.m)

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

%% Estimate LPV-VAR model
close all
clc

addpath(genpath('..\..\Core'))

T = 2*pi/OmegaRef;
ind_train = 2e3+(1:100*T*fs);
signals.response = ys(ind_train,1:5)';                                      % Response ( horizontal tower acceleration )
signals.scheduling_variables = beta(ind_train)';                            % Scheduling variable ( rotor azimuth )

na_max = 5;
pa_max = 15;

na = 1:na_max;
pa = 1:2:pa_max;

rss_sss = zeros(numel(na),numel(pa));
bic = zeros(numel(na),numel(pa));

for i = 1:numel(na)
    for j=1:numel(pa)

        structure.na = na(i);
        structure.pa = pa(j);

        options.basis.type = 'fourier';                                             % Type of functional basis
        options.basis.ind_ba = 1:structure.pa;

        M = estimate_lpv_var(signals,structure,options);
        rss_sss(i,j) = M.Performance.rss_sss;
        bic(i,j) = M.Performance.bic;

    end
end

%% Analysis of model order selection results
close all
clc

clr = parula(numel(pa));

na_opt = 2;

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
% xlim([2 inf])

nexttile
for i=1:numel(pa)
    plot(na,bic(:,i),'Color',clr(i,:),'LineWidth',1.5)
    hold on
end
xlabel('Model order')
ylabel('BIC')
grid on
% xlim([2 inf])

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

na_opt = 2;
pa_opt = 9;

structure.na = na_opt;
structure.pa = pa_opt;

options.basis.type = 'fourier';                                             % Type of functional basis
options.basis.ind_ba = 1:structure.pa;
options.estimator.type = 'ols';
options.estimator.Theta0 = zeros(n,n*na_opt*pa_opt);
options.estimator.Lambda0 = 25e-12*eye(n*na_opt*pa_opt);
options.estimator.V0 = cov(signals.response');

M = estimate_lpv_var(signals,structure,options);

m = 200;
Xi_range = linspace(0,1,m);
[Pyy,omega,omegaN,zeta] = MBA_lpv_var(M,Xi_range);

figure
tiledlayout(1,5)
for i=1:5
    nexttile
    imagesc(Xi_range,omega*fs/(2*pi),10*log10(squeeze(Pyy(i,i,:,:))))
    grid on
    axis xy
end


figure
tiledlayout(3,1)

nexttile
pwelch(signals.response',hann(2^10),2^10*3/4,2^10,fs)

nexttile(2,[2 1])
plot(omegaN'*fs/(2*pi),100*zeta','LineWidth',2)
xlim([0 fs/2])
grid on

figure
tiledlayout(1,2)

nexttile
plot(Xi_range,omegaN*fs/(2*pi),'LineWidth',2)
grid on
xlabel('Time (s)')
ylabel('Frequency (Hz)')

nexttile
plot(Xi_range,zeta*100,'LineWidth',2)
grid on
xlabel('Time (s)')
ylabel('Damping ratio (%)')


%%
close all
clc

Theta = reshape(M.Parameters.Theta,n,n*pa_opt,na_opt); 

SigmaTheta = kron( M.Parameters.SigmaTheta.K0, M.Parameters.SigmaTheta.SigmaW );
sigmaTh2 = diag(SigmaTheta);

tstat = abs( M.Parameters.Theta(:) ) ./ sqrt(sigmaTh2);
Tstat = reshape( tstat, n, n*pa_opt, na_opt );
tstat_threshold = sort(tstat,'ascend');

figure('Position',[100 100 600 300])
semilogx(tstat_threshold,sum(tstat>=tstat_threshold'),'-o','LineWidth',1.5)
grid on
xlabel('Threshold t-statistic')
ylabel('Non-zero coeffs.')

alpha = [0.0001 0.1 1];

cmap = [1 1 1; parula(20)];

figure
tiledlayout(3,2)
for j=1:3
    Mask = Tstat >= alpha(j);
    for i=1:na_opt
        nexttile
        imagesc(log10(abs(Theta(:,:,i)).*Mask(:,:,i)))
        colormap(cmap)
        % set(gca,'CLim',[0 inf])
    end
end

%%
close all
clc

m = 100;
Xi_range = linspace(0,1,m);
Mp = M;

figure('Position',[100 100 900 400])
tiledlayout(1,2)

for j=1:420

    Tstat = reshape( tstat, n, n, pa_opt, na_opt );
    Mask = Tstat >= tstat_threshold(j);
    Mp.a.projection = M.a.projection.*Mask;

    [Pyy,omega,omegaN,zeta] = MBA_lpv_var(Mp,Xi_range);

    nexttile(1)
    semilogx(tstat_threshold,sum(tstat>=tstat_threshold'),'-o','LineWidth',1.5)
    grid on
    xlabel('Threshold t-statistic')
    ylabel('Non-zero coeffs.')
    xline(tstat_threshold(j),'',['\alpha = ',num2str(tstat_threshold(j),'%2.2e')],'LabelVerticalAlignment','bottom')

    nexttile(2)
    % imagesc(log10(abs(reshape(Mp.a.projection,n,n*pa_opt*na_opt))))
    % colormap(cmap)
    imagesc(Xi_range,omega*fs/(2*pi),10*log10(squeeze(Pyy(1,1,:,:))))
    axis xy
    ylim([0 2.5]), set(gca,'CLim',[-100 -40])
    hold on
    plot(Xi_range,omegaN*fs/(2*pi),'Color',[1 1 1])
    hold off
    grid on


    % nexttile(4)
    % plot(omegaN'*fs/(2*pi),100*zeta')
    % ylim([-2 4])
    % xlim([0 2.5])
    % grid on

    % pause(0.01)
    drawnow
end

%% Analysis of residuals
close all
clc

figure
tiledlayout(2,n)
for i=1:n
    nexttile(i)
    normplot(M.Innovations(i,:))

    nexttile(i+n)
    histfit(M.Innovations(i,:))

end

figure
tiledlayout(n,2)
for i=1:n
    [xc,lags] = xcorr( M.Innovations(i,:), 40, 'coeff' );

    nexttile
    stem(lags,xc)

    nexttile
    pwelch(M.Innovations(i,:))

end