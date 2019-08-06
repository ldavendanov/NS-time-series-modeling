%--------------------------------------------------------------------------
% This script demonstrates the identification of the simulated vibration
% response of a wind turbine measured in the blade tip. The simulations
% were obtained with the wind turbine aeroelastic simulation package FAST
% [1]. Further details on the simulations are provided in [2].
% In the identification, it is assumed that the blade tip acceleration
% corresponds to the response of an LPV system where the scheduling
% variable is the instantaneous rotor azimuth.
% Identification is performed in discrete time by means of LPV-VAR models,
% in an output-only fashion, namely, only the response of the system
% (blade tip acceleration) is used in the identification.
%
% References : 
% [1] J. M. Jonkman & M. L. Buhl, "FAST user’s guide", Tech. rep. of the
%     National Renewable Energy Laboratory, U.S. Department ofEnergy,
%     Office of Energy Efficiency and Renewable Energy, Battelle, CO,
%     U.S.A. 2005.
% [2] L.D. Avendano-Valencia & S.D. Fassois, "Damage/fault diagnosis in an
%     operating wind turbine under uncertainty via a vibration response
%     Gaussian mixture random coefficient model based framework",
%     Mechanical Systems and Signal Processing, 91, pp. 326-353, 2017. DOI:
%     10.1016/j.ymssp.2016.11.028
% [3] L.D. Avendano-Valencia, E.N. Chatzi & S.D. Fassois, "In-operation
%     wind turbine modal analysis via LPV-VAR modeling", in Eds. Di Maio
%     and Castellini, "Rotating Machinery, Hybrid Test Methods,
%     Vibro-Acoustics % Laser Vibrometry: Proceedings of the 35th IMAC
%     2017", 8, 47-57, 2017.
% 
%
% Created by : David Avendano - July 2019
%--------------------------------------------------------------------------

%-- Clearing the workspace
clear
close all
clc

%% Loading the wind turbine signals

%-- Loading the wind turbine vibration data
load('Data\WT_Healthy1.mat','wtdata')

%-- Analyzing a single blade tip sensor on the flapwise and edgewise directions
t = wtdata.time;                                                            % Time vector
y = wtdata.blade_tip(:,1:2);                                                % Blade vibration (response variables)
xi = wtdata.azimuth/360;                                                    % Rotor azimuth (scheduling variable)

%-- Trimming the signal to remove initial transient
t0 = 1e3;                                                                   % Initial analysis time
N = 6e3;                                                                    % Signal length
t = t(1:N);
y = y(t0+(1:N),:);
xi = xi(t0+(1:N));

%-- Calculating the sampling period
Ts = t(2)-t(1);

%% Part 2 : Identification via LPV-AR models - Selection of the model order
% Standard model order selection is carried out. A range of model orders is
% proposed ( via 'na_max' ), while a plausible value of the functional
% basis order is selected ( 'pa' ). LPV-AR models are then calculated for
% various model structures within the range 1 up to na_max. Then different
% performance criteria are compared to determine the best model order.

close all
clc

%-- Fixing the input variables and training/validation indices
ind_train = false(1,N);
ind_train(1:4000) = true;                                                   % Indices for training of the LPV-AR model
ind_val = ~ind_train;                                                       % Indices for validation
signals.response = y(ind_train,:)';                                         % Response ( edgewise acceleration )
signals.scheduling_variables = xi(ind_train)';                              % Scheduling variable ( rotor azimuth )
sign_validation.response = y(ind_val,:)';                                   % Response ( edgewise acceleration )
sign_validation.scheduling_variables = xi(ind_val)';                        % Scheduling variable ( rotor azimuth )

%-- Structural parameters of the LPV-AR model
na_max = 25;                                                                % Maximum model order
pa = 7;                                                                     % Functional basis order
options.basis.type = 'fourier';                                             % Type of functional basis

%-- Initializing computation matrices
rss_sss = zeros(2,na_max);                                                  % Residual sum of squares over series sum of squares
lnL = zeros(2,na_max);                                                      % Log likelihood
bic = zeros(1,na_max);                                                      % Bayesian Information Criterion

%-- Model order selection loop
for na = 1:na_max
    
    %-- Calculating training performance criteria
    order = [na pa];
    M = estimate_lpv_var(signals,order,options);
    rss_sss(1,na) = M.performance.rss_sss;
    lnL(1,na) = M.performance.lnL;
    bic(na) = M.performance.bic;
    
    %-- Calculating validation performance criteria
    [~,criteria] = simulate_lpv_var(sign_validation,M);
    rss_sss(2,na) = criteria.rss_sss;
    lnL(2,na) = criteria.lnL;
end

%-- Plotting performance criteria
figure('Position',[100 100 1200 400])
subplot(131)
semilogy(2:na_max,rss_sss(:,2:na_max)*100)
grid on
xlabel('Model order')
ylabel('RSS/SSS [%]')
legend({'Training','Validation'})

subplot(132)
plot(2:na_max,lnL(:,2:na_max))
grid on
xlabel('Model order')
ylabel('Log-likelihood')
legend({'Training','Validation'})

subplot(133)
plot(2:na_max,bic(2:na_max))
grid on
xlabel('Model order')
ylabel('BIC')

figure('Position',[100 600 800 400])
bins = linspace(0,1,40);
histogram(xi(ind_train),bins)
hold on
histogram(xi(ind_val),bins)
grid on
legend({'Training','Validation'})
xlabel('Scheduling variable \xi')

%% Part 3 : Identification via LPV-AR models - Basis order analysis
% Based on the model order selected on Part 2, here an analysis of the
% coefficients of the LPV-AR model is carried out. The objective of this
% analysis is to determine if some coefficients may be deemed as zero. For
% that purpose a hypothesis test is performed by assuming that the
% coefficients are Gaussian distributed with zero mean and covariance
% determined by the estimation algorithm.
% If the coefficients are lower than the threshold for a specific Type I
% error probability, then with such probability the coefficient can be
% deemed as equal to zero.
% If a the coefficients of a basis are consistently equal to zero, then the
% respective basis may be rejected from the model, thus leading to a more
% compact model. 

close all
clc

clear lnL rss_sss bic

%-- Structural parameters of the LPV-AR model
na = 12;                                                                    % Model order
pa = 7;                                                                     % Functional basis order
order = [na pa];                                                            % Order parameters
options.basis.type = 'fourier';                                             % Type of functional basis

%-- Estimating the LPV-AR model for the training data
M = estimate_lpv_var(signals,order,options);
rss_sss(1,1) = M.performance.rss_sss;
lnL(1,1) = M.performance.lnL;

%-- Validating the LPV-AR model on the validation data
[xhat0,criteria] = simulate_lpv_var(sign_validation,M);
rss_sss(2,1) = criteria.rss_sss;
lnL(2,1) = criteria.lnL;

%-- Chi square test for the LPV-VAR coefficients
% This test evaluates the hypothesis that the LPV-VAR coefficients are zero.
% For that purpose, it is assumed that each coefficient is Gaussian
% distributed with mean zero and covariance equal to that provided by the
% estimation algorithm.

chi2_theta = reshape( M.performance.chi2_theta, 2, 2, pa, na );             % Test statistic (chi squared distributed)
alph_chi2 = 10.^(-4:-1);                                                    % Probability of type I error (rejecting the null hypothesis)
rho = chi2inv( 1-alph_chi2, 1 );                                            % Threshold for error probablity

Mrk = 'oxd^s*o';

figure('Position',[100 100 600 800])
pt = zeros(pa,1);
for j=1:pa
    pt(j) = semilogy( 1:na, squeeze(chi2_theta(1,1,j,:)), ['--',Mrk(j)], 'MarkerSize', 10, 'LineWidth', 2 );
    hold on
end

grid on
for i=1:numel(rho)
    semilogy( [1 na], rho(i)*[1 1], 'k' )
    text( na-0.1, 1.25*rho(i), ['\alpha = ',num2str(alph_chi2(i))],'HorizontalAlignment','right' )
end
set(gca,'XTick',1:na)
legend(pt, {'$f_0(\xi)$','$f_1(\xi)$','$f_2(\xi)$','$f_3(\xi)$','$f_4(\xi)$','$f_5(\xi)$','$f_6(\xi)$'},'Interpreter','latex')
xlabel('Model order')
ylabel('\chi^2 test statistic')

%-- Selected basis indices
basis_indices = squeeze(max( max( max( chi2_theta > rho(2) ,[],4), [], 1), [], 2 ));

%% Validating the obtained LPV-AR model in the validation data
close all
clc

opts = options;
opts.basis.indices = basis_indices;

%-- Estimating the LPV-AR model for the training data
M1 = estimate_lpv_var(signals,order,opts);
rss_sss(1,2) = M1.performance.rss_sss;
lnL(1,2) = M1.performance.lnL;

%-- Validating the LPV-AR model on the validation data
[xhat,criteria] = simulate_lpv_var(sign_validation,M1);
rss_sss(2,2) = criteria.rss_sss;
lnL(2,2) = criteria.lnL;

figure('Position',[100 100 900 600])
for i=1:2
    subplot(2,1,i)
    plot(t(ind_val),xhat0(i,:),t(ind_val),xhat(i,:),t(ind_val),y(ind_val,i))
    grid on
    xlabel('Time [s]')
    if i==1
        ylabel('Acceleration - Flapwise [m/s^2]')
    else
        ylabel('Acceleration - Edgewise [m/s^2]')
    end
    legend({'Complete LPV-AR model','Reduced LPV-AR model','Original'})
end

figure('Position',[1000 100 900 800])
A0 = M.a;
AA = zeros(size(A0));
AA(:,:,basis_indices,:) = M1.a;
for i=1:pa
    subplot(pa,1,i)
    plot(squeeze(A0(1,1,i,:)),['--',Mrk(1)])
    hold on
    if basis_indices(i)
        plot(squeeze(AA(1,1,i,:)),['--',Mrk(1)])
        legend({'Complete','Reduced'})
    else
        legend('Complete')
    end
    grid on
    ylabel(['$a_{i,',num2str(i),'}$'],'Interpreter','latex')
    xlabel('AR order')
    set(gca,'XTick',1:na)
end

%% Model based analysis
% Analysis of the dynamics of the identified LPV-AR model
close all
clc

m = 100;
Xi_range = linspace(0,1,m);
[Pyy,omega,omegaN,zeta,Psi] = MBA_lpv_var(M1,Xi_range);

figure('Position',[100 100 900 800])

for i=1:2
    subplot(2,1,i)
    imagesc(Xi_range,omega/(2*pi*Ts),10*log10(squeeze(Pyy(i,i,:,:))))
    axis xy
    xlabel('Rotor azimuth [rad/2\pi]')
    ylabel('Frequency [Hz]')
    grid on
    cbar = colorbar;
    cbar.Label.String = 'PSD [dB]';
    set(gca,'YTick',0:1:1/(2*Ts),'XTick',0:0.25:1)
    ylim([0 1/(2*Ts)])
    
end


figure('Position',[1000 100 900 480])
dx = 0.01*[0 1];
dy = 0.04*[0 1];
for i=1:2*na
    for j=1:m
        if zeta(i,j) <= 0.5 && zeta(i,j) >= 0
            imagesc(Xi_range(j)+dx,omegaN(i,j)/(2*pi*Ts)+dy,100*zeta(i,j))
            hold on
        end
    end
end
xlabel('Rotor azimuth [rad/2\pi]')
ylabel('Frequency [Hz]')
xlim([0 1])
ylim([0 1/(2*Ts)])
axis xy
grid on
cbar = colorbar;
cbar.Label.String = 'Damping ratio [%]';
set(gca,'XTick',0:0.25:1)
