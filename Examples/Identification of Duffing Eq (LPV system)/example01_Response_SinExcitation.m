%--------------------------------------------------------------------------
% This script provides the analysis of the response of a non-linear
% oscillator defined by the Duffing equation:
%
%       y'' + delta*y' + alpha*y + beta*y^3 = gamma'F(t)
%
% when the excitation force F(t) = cos( omega*t ).
%
% Created by : David Avendano - July 2019
%--------------------------------------------------------------------------

%-- Cleaning the workspace
clear
close all
clc

%% Setting up the system properties

%-- Parameters of the Duffing equation
alpha = 1;                                                                  % Stiffness parameter (linear)
beta = 4e-1;                                                                % Stiffness parameter (cubic)
delta = 0.1;                                                                % Damping term
gamma = 2;                                                                  % Input gain
theta = [alpha beta gamma delta];                                           % Parameter vector

%% Calculating the theoretical frequency response
close all
clc

%-- Analytical form of the frequency response of the Duffing oscillator
Omega = 8;                                                                  % Maximum frequency of the excitation
g = @(omega,z) gamma^2 - ( ( omega.^2-alpha-3/4*beta*z.^2 ).^2 + ( delta*omega ).^2 )*z.^2;

fimplicit(g,[0 Omega 0 8])
grid on
xlabel('Excitation frequency (rad/s)')
ylabel('Displacement amplitude')


%% Plotting the potential energy of the system as a function of amplitude

A = linspace(-10,10);                                                       % Vibration amplitude
K = alpha + beta*A.^2;                                                      % Linearized stiffness

figure('Position',[100 100 1200 400])
subplot(131)
plot(A,K)
ylabel('Stiffness')
xlabel('Amplitude')
grid on

subplot(132)
plot(A,K.*A)
ylabel('Restoring (spring) force')
xlabel('Amplitude')
grid on

subplot(133)
plot(A,K.*A.^2)
ylabel('Potential energy')
xlabel('Amplitude')
grid on

%% Frequency sweep
% The response of the non-linear system is calculated for stationary input
% with different frequencies

%-- Simulation parameters
fs = 500;                                                                   % Sampling frequency
T = 400;                                                                    % Analysis period (s)
t = linspace(0,T,T*fs);                                                     % Time vector (s)
x0 = [0 0]';                                                                % Initial state of the system

%-- Excitation properties
Nf = 400;                                                                   % Number of frequencies
Omega = 8;                                                                  % Maximum frequency of the excitation
omega = linspace(0,Omega,Nf);                                               % Vector with the excitation frequencies

%-- Initializing computation matrices
Y = zeros(T*fs,Nf);                                                         % Matrix with displacement response
V = zeros(T*fs,Nf);                                                         % Matrix with velocity response

%-- Calculation loop
parfor i=1:Nf
    F = @(t) cos(omega(i)*t);                                               % Excitation signal - cosine at the specified frequency
    [~,y] = ode45( @(t,y)DuffingEq(t,y,F,theta), t, x0 );                   % Integrating the non-linear system
    Y(:,i) = y(:,1);                                                        % Extracting the displacement response
    V(:,i) = y(:,2);                                                        % Extracting the velocity response
end

%% Plotting results of the frequency sweep
close all
clc

%-- Calculating the PSD of the displacement response
Nwin = 2^14;                                                                % Window size (samples)
Pyy = zeros(Nwin/2+1,Nf);                                                   % Power Spectral Density matrix
for i=1:Nf
    [Pyy(:,i),f] = pwelch(Y(fs*T/25:end,i),hamming(Nwin),Nwin/4,Nwin,fs);   % Calculating the PSD (initial 25% of the signal removed to reduce transient effects)
end

%-- PSD as a function of the excitation frequency
figure('Position',[100 100 900 800])
imagesc(omega,2*pi*f,10*log10(Pyy))
axis xy
hold on
plot([0 Omega],[1 1]*sqrt(alpha),'--k','LineWidth',1)
plot(sqrt(alpha)*[1 1],[0 2*pi*fs/2],'--k','LineWidth',1)
ylim([0 Omega])
xlabel('Excitation frequency (rad/s)')
xlabel('Response frequency (rad/s)')

%-- Excitation and response plots
figure('Position',[100 100 900 800])
D_omega = 20;
for i=1:8
    subplot(4,2,i)
    plot(t,gamma*cos(omega(D_omega*i)*t))
    hold on
    plot(t,Y(:,D_omega*i))
    xlim([150 250])
    ylim(1.5*gamma*[-1 1])
    grid on
    
    xlabel('Time [s]')
    ylabel('Displacement')
    title(['\omega = ',num2str(omega(D_omega*i),'%2.2f'),' rad/s'])
    
end

%-- Phase plots
figure('Position',[1000 100 900 800])
for i=1:8
    subplot(4,2,i)
    plot(Y(:,D_omega*i),V(:,D_omega*i))
    title(['\omega = ',num2str(omega(D_omega*i),'%2.2f'),' rad/s'])
    grid on
    xlabel('Displacement')
    ylabel('Velocity')
    xlim(1.5*gamma*[-1 1])
    ylim(5*[-1 1])
end

%-- Frequency response plot
figure('Position',[1000 100 600 400])
z = abs(hilbert(Y));                                                        % Vibration amplitude obtained from the Hilbert transform
fimplicit(g,[0 Omega 0 8],'k','LineWidth',2)
hold on
plot(omega,mean(z((T*fs*0.4:T*fs),:)),'.r','LineWidth',2,'MarkerSize',8)
grid on
xlabel('Excitation frequency (rad/s)')
ylabel('Displacement amplitude')
legend({'Analytical','Experimental'})