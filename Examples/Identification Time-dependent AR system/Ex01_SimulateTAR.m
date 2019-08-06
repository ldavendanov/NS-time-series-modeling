%--------------------------------------------------------------------------
% This script simulates a TAR model for given parameter trajectories.
% The parameter trajectories are created around a mean pole location with
% magnitude 0.9 and angle 2*pi*15/fs [rad], where fs = 100 [Hz] is the 
% sampling frequency. The parameter modulation frequency is of 0.1 [Hz].
%
% The script creates the parameter trajectories, calls a function to
% simulate the TAR model and plots the obtained time series, parameter
% trajectories and locus of the pole trajectories.
%
% Created by :  David Avendano - January 2015 - Ver 1.0
% Updated :     David Avendano - July 2019    - Ver 1.1
%--------------------------------------------------------------------------
clear
close all
clc

%% Part 0 : Setting the simulation parameters -----------------------------

N = 8000;                                                                   % Number of samples
t = 1:N;                                                                    % Time vector
fs = 100;                                                                   % Sampling frequency

%% Part 1 : Simulating the output of a simple TAR model

%-- Creating the parameter trajectories of the TAR model ------------------
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

%-- Defining a time-dependent innovations variance ------------------------
sw = 0.25*(1+0.2*sin(omegaF*t+pi*rand(1)));                                 % Innovations variance

%-- Simulating a realization of the process -------------------------------
[y,w,criteria] = SimulateTARMA(a,[],sw);

%-- Showing the results ---------------------------------------------------
figure('position',[50 100 1200 650])
subplot(3,2,1)
plot(t/fs,y)
set(gca,'YLim',[-5 5])
ylabel('Time series')

subplot(3,2,3)
plot(t/fs,a)
ylabel('TAR parameters')
legend('a_1[t]','a_2[t]')

subplot(3,2,5)
plot(t/fs,sw)
set(gca,'YLim',[0 0.6])
ylabel('Innovations variance')
xlabel('Time [s]')

subplot(3,2,[4 6])
plot(real(rho)',imag(rho)','.')
zgrid
axis square
xlabel('Real')
ylabel('Imaginary')
title('Pole locus')