%--------------------------------------------------------------------------
% This script simulates a TAR model for given parameter trajectories and
% uses the Smoothness Priors TAR and Generalized Stochastic Constraint TAR
% methods to compute estimates of the parameter trajectories.
%
% The script "Ex01_SimulateTAR.m" explains the simulation model used in
% this example
%
% Created by : David Avendano - January 2015
%   Revised  : David Avendano - January 2022
%--------------------------------------------------------------------------
clear
close all
clc

% Adding into the path the functions corresponding to simulation of TARMA models
addpath(genpath('..\..\Core\'))

%% Creating a realization of a TAR process --------------------------------

fprintf('Simulating TAR model...\n')

%-- Simulation parameters
N = 8000;                                                                   % Number of samples
t = 1:N;                                                                    % Time vector
fs = 100;                                                                   % Sampling frequency

%-- Creating the parameter trajectories of the TAR model 
r0 = 0.8;                                                                   % Mean pole magnitude
omega0 = 2*pi*15/fs;                                                        % Mean pole angle
omegaF = 2*pi*0.1/fs;                                                       % Frequency of modulation of the parameters
r = r0 + 0.05*cos(omegaF*t);                                                % Trajectory of the pole magnitude
omega = omega0 + 2*pi*2/fs*sin(omegaF*t);                                   % Trajectory of the pole angle
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

%% Estimating the model parameters via UPE-TARMA methods ------------------

%-- Estimating the parameter trajectories using smoothness priors
fprintf('Estimating via smoothness priors method...\n')
order = 2;                                                                  % Model order
[a_SP,~,sw_SP,otherSP] = sptarma(y,order);

%-- Estimating the parameter trajectories using smoothness priors
fprintf('Estimating via generalized stochastic constraint method...\n')
order = 2;                                                                  % Model order
[a_GSC,~,sw_GSC,otherGSC] = gsctarmaEM(y,order);

%% Plotting the results ---------------------------------------------------
close all
clc

figure('position',[50 100 900 750])
subplot(311)
plot(t/fs,a(1,:),'k','LineWidth',2)
hold on
plot(t/fs,a_SP(1,:),t/fs,a_GSC(1,:))
set(gca,'YLim',[-1.5 0])
ylabel('a_1[t]')
xlabel('Time [s]')
legend('Original model','SP-TAR estimates','GSC-TAR estimates')

subplot(312)
plot(t/fs,a(2,:),'k','LineWidth',2)
hold on
plot(t/fs,a_SP(2,:),t/fs,a_GSC(2,:))
set(gca,'YLim',[0 1])
ylabel('a_2[t]')
xlabel('Time [s]')
legend('Original model','SP-TAR estimates','GSC-TAR estimates','location','SouthEast')

subplot(313)
plot(t/fs,sw,'k','LineWidth',2)
hold on
plot(t/fs,sw_SP,t/fs,sw_GSC)
ylabel('\sigma_w^2[t]')
xlabel('Time [s]')
legend('Original model','SP-TAR estimates','GSC-TAR estimates','location','SouthEast')