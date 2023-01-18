%--------------------------------------------------------------------------
% This script simulates a TAR model for given parameter trajectories and
% uses the Functional Series TAR method to compute estimates of the
% parameter trajectories.
%
% The script "Ex01_SimulateTAR.m" explains the simulation model used in
% this example
%
% Created by :  David Avendano - January 2015 - Ver. 1.0
% Updated :     David Avendano - July 2019    - Ver. 1.1
%--------------------------------------------------------------------------
clear
close all
clc

addpath(genpath('..\..\Core\'))

%% Creating a realization of a TAR process --------------------------------

fprintf('Simulating TAR model...\n')

%-- Simulation parameters
N = 8000;                                                                   % Number of samples
t = 1:N;                                                                    % Time vector
fs = 100;                                                                   % Sampling frequency

%-- Creating the parameter trajectories of the TAR model 
r0 = 0.94;                                                                  % Mean pole magnitude
omega0 = 2*pi*15/fs;                                                        % Mean pole angle
omegaF = 2*pi*0.1/fs;                                                       % Frequency of modulation of the parameters
r = r0 + 0.04*cos(omegaF*t);                                                % Trajectory of the pole magnitude
omega = omega0 + 2*pi*4/fs*sin(omegaF*t);                                   % Trajectory of the pole angle
rho = [r.*exp(1i*omega); r.*exp(-1i*omega)];                                % Pole trajectories
a = zeros(2,N);                                                             % TAR parameter trajectories
for tt=1:N
    a0 = poly(rho(:,tt));
    a(:,tt) = a0(2:end)';
end

%-- Defining the innovations variance
sw = 0.25*(1+0.2*sin(omegaF*t+pi*rand(1)));                                 % Innovations variance

%-- Simulating a realization of the process 
[y,w,criteria] = SimulateTARMA(a,[],sw);

%% Estimating the model parameters via FS-TAR methods ---------------------
close all
clc

%-- Defining the response signal segment
signals.response = y;

%-- Defining the common model structure
structure.na = 2;                                                           % Model order (common for both models)

%-- Estimating the parameter trajectories using FS-TAR model with sinusoidal basis
fprintf('Estimating via FS-TAR model with sinusoidal basis...\n')

structure.pa = 17;                                                          % Basis order : AR parameters
structure.ps = 17;                                                          % Basis order : Innovations variance
options.basis.type = 'sinus';                                               % Basis type : sinusoidal
options.basis.ind_ba = [1 16 17];                                           % Basis indices : AR parameters
options.basis.ind_bs = [1 16 17];                                           % Basis indices : innovations variance
options.estimator.type = 'multi-stage';
options.VarEstim.Type = 'iv';                                               % Innovations variance estimator (instantaneous variance)

M1 = estimate_fs_tar(signals,structure,options);                            % Estimate the FS-TAR model

%-- Estimating the parameter trajectories using FS-TAR model with b-spline basis
fprintf('Estimating via FS-TAR model with b-spline basis...\n')

structure.pa = 26;                                                          % Basis order
options.basis.type = 'bspline';                                             % Basis type : b-splines
options.basis.ind_ba = 1:structure.pa;                                      % Basis indices : AR parameters
options.basis.ind_bs = 1:structure.pa;                                      % Basis indices : innovations variance
options.VarEstim.Type = 'iv';                                               % Innovations variance estimator (instantaneous variance)

M2 = estimate_fs_tar(signals,structure,options);                            % Estimate the FS-TAR model


%% Plotting the results ---------------------------------------------------
close all
clc

figure('position',[50 100 900 750])
for i=1:2
    subplot(3,1,i)
    plot(t/fs,a(i,:),'k','LineWidth',2)
    hold on
    plot(t/fs,M1.ar_part.a_time(i,:),t/fs,M2.ar_part.a_time(i,:))
    ylabel(['$a_',num2str(i),'[t]$'],'Interpreter','latex')
    xlabel('Time [s]')
    legend('Original model','FS-TAR sinusoidal','FS-TAR bspline')
    grid on
    
end

subplot(313)
plot(t/fs,sw,'k','LineWidth',2)
hold on
plot(t/fs,M1.InnovationsVariance.sigmaW2,...
     t/fs,M2.InnovationsVariance.sigmaW2)
set(gca,'YLim',[0 0.4])
ylabel('\sigma_w^2[t]')
xlabel('Time [s]')
legend('Original model','FS-TAR sinusoidal','FS-TAR bspline','location','SouthEast')
grid on