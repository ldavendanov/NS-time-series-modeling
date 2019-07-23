%--------------------------------------------------------------------------
% This script demonstrates the identification of a non-linear system
% determined by the Duffing equation:
%
%       y'' + delta*y' + alpha*y + beta*y^3 = gamma'F(t)
%
% when the excitation force F(t) corresponds to a single sinusoid with
% variable frequency (chirp). The excitation is designed so that the
% frequency of the sinusoid crosses upwards and downwards the non-linear
% resonance so as to trigger the hysteresis effect on the frequency
% response of the system. 
% In the identification, it is assumed that the system is LPV where the
% scheduling variable is the displacement variable, so that:
%
%       y'' + a1*y' + a2(t)*y = gamma'F(t)
%
% where:
%
%       a1 = delta   and a2(t) = (alpha* + beta*y^2)
%
% Identification is performed in discrete time by means of LPV-ARX models
% in an excitation-response setting.
%
% Created by : David Avendano - July 2019
%--------------------------------------------------------------------------

%-- Clearing the workspace
clear
close all
clc

%% Part 1 : Creating a simulation of the system's response to a chirp excitation

%-- Simulation parameters
T = 400;                                                                    % Analysis period (s)
fs0 = 128;                                                                  % Sampling frequency for simulation
N = T*fs0;                                                                  % Number of samples
t = linspace(1/fs0,T,N);                                                    % Time vector (s)

%-- Creating the excitation force
f0 = 0;
f1 = fs0/8;
t1 = T/2;
F = @(t)sf_cosine(t,f0,t1,f1);                                              % Creating the excitation based on the provided NID sample

%-- Parameters of the non-linear system
alpha = (2*pi*2)^2;                                                         % Stiffness parameter (linear)
beta = 80;                                                                  % Stiffness parameter (cubic)
delta = 2*0.01*sqrt(alpha);                                                 % Damping parameter (1% damping ratio)
gamma = 100;                                                                % Input gain
theta = [alpha beta gamma delta];                                           % Parameter vector
x0 = [0 0]';                                                                % Initial state of the system

%-- Initializing computation matrices
[~,y] = ode45( @(t,y)DuffingEq(t,y,F,theta), t, x0 );                       % Integrating the non-linear system
force = zeros(1,N);
for i=1:N
    force(i) = F(t(i));
end

%% Part 1b : Resampling the signal
% Signal is resampled into the bandwidth of higher response power. Correct
% setting is essential for accurate identification of the non-linear
% system.

close all
clc

%-- Properties of the downsampled signal
N = 25600;                                                                  % Signal length
T0 = 0;                                                                     % Time to start analysis
fs = 64;                                                                    % Analysis sampling frequency

%-- Resampling
x = resample(y(:,1),fs,fs0);                                                % Resampling the response signal
u = resample(force,fs,fs0)';                                                % Resampling the excitation signal
t = t(1:fs0/fs:end);                                                        % Resampling the time vector

%-- Trimming the signals
x = x(T0+(1:N));                                                            % Trimming the response signal into the desired analysis period
u = u(T0+(1:N));                                                            % Trimming the excitation signal into the desired analysis period
t = t(T0+(1:N));                                                            % Trimming the time vector into the desired analysis period

%% Part 2 : Identification via LPV-AR models - Selection of the model order
% Standard model order selection is carried out. A range of model orders is
% proposed ( via 'na_max' ), while a plausible value of the functional
% basis order is selected ( 'pa' ). LPV-AR models are then calculated for
% various model structures within the range 1 up to na_max. Then different
% performance criteria are compared to determine the best model order.

close all
clc

%-- Fixing the input variables and training/validation indices
signals.excitation = u(:)';                                                 % Excitation (force)
signals.response = x(:)';                                                   % Response (displacement)
signals.scheduling_variables = x(:)';                                       % Scheduling variable (also displacement!)

ind_train = true(1,N);                                                      % Indices for training of the LPV-AR model
ind_val = ind_train;                                                        % Indices for validation

%-- Structural parameters of the LPV-AR model
na_max = 20;                                                                % Maximum model order
pa = 4;                                                                     % Functional basis order
options.basis.type = 'hermite';                                             % Type of functional basis

%-- Initializing computation matrices
rss_sss = zeros(2,na_max);                                                  % Residual sum of squares over series sum of squares
lnL = zeros(2,na_max);                                                      % Log likelihood
bic = zeros(1,na_max);                                                      % Bayesian Information Criterion

%-- Model order selection loop
for na = 1:na_max
    
    %-- Calculating training performance criteria
    order = [na na pa];
    M = estimate_lpv_arx(signals,order,options);
    rss_sss(1,na) = M.performance.rss_sss;
    lnL(1,na) = M.performance.lnL;
    bic(na) = M.performance.bic;
    
    %-- Calculating validation performance criteria
    [~,criteria] = simulate_lpv_arx(signals,M);
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
na = 12;                                                                     % Model order
pa = 4;                                                                     % Functional basis order
order = [na na pa];                                                         % Order parameters
options.basis.type = 'hermite';                                             % Type of functional basis

%-- Estimating the LPV-AR model for the training data
M = estimate_lpv_arx(signals,order,options);
rss_sss(1,1) = M.performance.rss_sss;
lnL(1,1) = M.performance.lnL;

%-- Validating the LPV-AR model on the validation data
[xhat0,criteria] = simulate_lpv_arx(signals,M);
rss_sss(2,1) = criteria.rss_sss;
lnL(2,1) = criteria.lnL;

%-- Chi square test for the LPV-AR coefficients
% This test evaluates the hypothesis that the LPV-AR coefficients are zero.
% For that purpose, it is assumed that each coefficient is Gaussian
% distributed with mean zero and covariance equal to that provided by the
% estimation algorithm.

chi2_theta = M.performance.chi2_theta;                                      % Test statistic (chi squared distributed)
chi2_theta = reshape(chi2_theta,pa,2*na+1);
alph_chi2 = 10.^(-4:-1);                                                    % Probability of type I error (rejecting the null hypothesis)
rho = chi2inv( 1-alph_chi2, 1 );                                              % Threshold for error probablity

Mrk = 'oxd^';

figure('Position',[100 100 600 800])
pt = zeros(pa,1);
for j=1:pa
    pt(j) = semilogy( 1:2*na+1, chi2_theta(j,:), ['--',Mrk(j)], 'MarkerSize', 10, 'LineWidth', 2 );
    hold on
end

grid on
for i=1:numel(rho)
    semilogy( [1 2*na+1], rho(i)*[1 1], 'k' )
    text( 1.1, 1.2*rho(i), ['\alpha = ',num2str(alph_chi2(i))] )
end
set(gca,'XTick',1:2*na+1)
legend(pt, {'$f_0(\xi)$','$f_1(\xi)$','$f_2(\xi)$','$f_3(\xi)$'},'Interpreter','latex')
xlabel('Model order')
ylabel('\chi^2 test statistic')

%-- Selected basis indices
basis_indices = max( chi2_theta > rho(3) ,[],2);

%% Validating the obtained LPV-AR model in the validation data
close all
clc

figure('Position',[100 100 900 800])
subplot(211)
plot(t,xhat0,t,x)
% xlim([30 60])
grid on
xlabel('Time (s)')
ylabel('Displacement')
legend({'LPV-AR model','Original'})

subplot(212)
pwelch([xhat0;x']')
legend({'LPV-AR model','Original'})
