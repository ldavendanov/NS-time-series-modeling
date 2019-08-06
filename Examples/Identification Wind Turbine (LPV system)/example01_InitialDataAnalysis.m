%--------------------------------------------------------------------------
% This script provides an initial exploration into the simulated vibration
% response of a wind turbine measured in the blade tip. The simulations
% were obtained with the wind turbine aeroelastic simulation package FAST
% [1]. Further details on the simulations are provided in [2].
% Here the wind turbine vibration plot is displayed in time and frequency
% domain.
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
%
% Created by : David Avendano - July 2019
%--------------------------------------------------------------------------

clear
close all
clc

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

%-- Time domain plots -----------------------------------------------------
figure('Position',[100 100 900 600])
subplot(211)
plot(t,y)
xlim([50 100])
grid on
xlabel('Time [s]')
ylabel('Acceleration [m/s^2]')
legend({'Flapwise','Edgewise'})

subplot(212)
plot(t,xi)
xlim([50 100])
grid on
xlabel('Time [s]')
ylabel('Rotor azimuth [rad/2\pi]')

%-- Spectral analysis -----------------------------------------------------
[Pyy,f] = pwelch(y,hamming(1024),512,1024,1/Ts);
[Syy1,ff,tt] = spectrogram(y(:,1),hamming(512),510,512,1/Ts);
Syy2 = spectrogram(y(:,2),hamming(512),510,512,1/Ts);


figure('Position',[1000 100 900 400])
subplot(141)
plot(10*log10(Pyy(:,1)),f)
set(gca,'XDir','reverse')
ylabel('Frequency [Hz]')
xlabel('PSD [dB]')
grid on

subplot(1,4,2:4)
imagesc(tt,ff,10*log10(abs(Syy1)))
axis xy
xlabel('Time [s]')
cbar = colorbar;
cbar.Label.String = 'PSD [dB]';
grid on
title('Flapwise acceleration')

figure('Position',[1000 550 900 400])
subplot(141)
plot(10*log10(Pyy(:,2)),f)
set(gca,'XDir','reverse')
ylabel('Frequency [Hz]')
xlabel('PSD [dB]')
grid on

subplot(1,4,2:4)
imagesc(tt,ff,10*log10(abs(Syy2)))
axis xy
xlabel('Time [s]')
cbar = colorbar;
cbar.Label.String = 'PSD [dB]';
grid on
title('Edgewise acceleration')