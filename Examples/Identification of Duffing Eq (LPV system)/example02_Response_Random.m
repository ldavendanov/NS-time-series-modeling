%--------------------------------------------------------------------------
% This script provides the analysis of the response of a non-linear
% oscillator defined by the Duffing equation:
%
%       y'' + delta*y' + alpha*y + beta*y^3 = gamma'F(t)
%
% when the excitation force F(t) is a normally and identically distributed
% noise.
%
% Created by : David Avendano - July 2019
%--------------------------------------------------------------------------

%-- Clearing the workspace
clear
close all
clc

%-- Simulation parameters
T = 800;                                                                    % Analysis period (s)
fs = 128;                                                                   % Sampling frequency
N = T*fs;                                                                   % Number of samples
t = linspace(1/fs,T,N);                                                     % Time vector (s)

%-- Creating the excitation force
f = randn(1,N);                                                             % Sampling from a normally random variable
F = @(t) NIDexcitation(t,f,fs);                                             % Creating the excitation based on the provided NID sample

%-- Parameters of the non-linear system
alpha = (2*pi*2)^2;                                                         % Stiffness parameter (linear)
beta = 80;                                                                  % Stiffness parameter (cubic)
delta = 2*0.01*sqrt(alpha);                                                 % Damping parameter (1% damping ratio)
x0 = [0 0]';                                                                % Initial state of the system

%-- Initializing computation matrices
Y = zeros(N,12);                                                            % Matrix with displacement responses


%% Calculating the response for increasing excitation amplitude

%-- Computation loop
parfor i=1:12
    
    gamma = 10^(-3+i/2);                                                    % Input gain
    theta = [alpha beta gamma delta];                                       % Parameter vector
    
    [~,y] = ode45( @(t,y)DuffingEq(t,y,F,theta), t, x0 );                   % Integrating the non-linear system
    Y(:,i) = y(:,1);                                                        % Extracting the displacement response
end

%% Plotting results
close all
clc

clr = lines(4);

%-- Excitation and response plots for the maximum input gain
figure('Position',[100 100 900 400])
subplot(2,4,2:4)
plot(t,f)
grid on
xlim([280 300])
xlabel('Time [s]')
ylabel('Excitation')

subplot(2,4,6:8)
plot(t,Y(:,12))
grid on
xlim([280 300])
yl = get(gca,'YLim');
xlabel('Time [s]')
ylabel('Response')

subplot(2,4,5)
A = linspace( yl(1), yl(2) );
plot( alpha + beta.*A.^2, A )
set(gca,'XDir','reverse')
xlabel('Stiffness')
ylabel('Amplitude')
grid on

%-- Response at different levels of excitation gain
figure('Position',[100 100 1200 900])
for i=1:6
    subplot(6,2,13-2*i)
    gamma = 10^(-3+(2*i)/2);
    plot(t,Y(:,2*i)/gamma)
    xlim([100 120])
    ylim(2.5e-2*[-1 1])
    grid on
    
    legend(['\gamma = ',num2str(gamma)])
    if i==1, xlabel('Time [s]'), end
    
end

subplot(6,2,2*(1:6))
Nfrec = 2^10;
[Pyy,frec] = pwelch(Y(:,2:2:end),hamming(Nfrec),Nfrec/2,Nfrec,fs);

for i=1:6
    gamma = 10^(-3+(2*i)/2);
    plot(2*pi*frec,10*log10(Pyy(:,i)),'Color',clr(1,:))
    hold on
    text(195,10*log10(Pyy(floor(Nfrec*200/(2*pi*fs)),i))-2,['\gamma = ',num2str(gamma)],'HorizontalAlignment','right')
end
grid on
xlim([0 200])

yl = get(gca,'YLim');
plot(sqrt(alpha)*[1 1],yl,'--k')
text(1.5*sqrt(alpha),-150,'$\omega = \sqrt{\alpha}$','Interpreter','latex')

xlabel('Frequency [rad/s]')
ylabel('Power [dB]')


%% Frozen analysis of the dynamics
% The assumption is that the dynamics remain linear at a given vibration
% amplitude. Although this is assumption is useless in practice, since the
% vibration amplitude changes instantaneously, it helps to understand the
% point when the non-linear behaviour is triggered.

close all
clc

%-- System parameters
alpha = (2*pi*2)^2;                                                         % Stiffness parameter (linear)
beta = 80;                                                                  % Stiffness parameter (cubic)
delta = 2*0.01*sqrt(alpha);                                                 % Damping ratio
gamma = 1;                                                                  % Input gain

%-- Simulation parameters
Na = 500;                                                                   % Number of vibration amplitudes
A = logspace(-4,1,Na);                                                      % Vibration amplitude vector (log-scale)
Nfft = 512;                                                                 % Number of frequency points
om = linspace(0,100,Nfft);                                                  % Frequency vector

H = zeros(Nfft,Na);                                                         % Matrix of frequency responses
for i=1:Na
    H(:,i) = freqs(delta,[1 delta alpha+beta*A(i)^2],om);                   % Frozen FRF at the specific vibration amplitude
end

%-- Plotting results
figure('Position',[100 100 900 450])

subplot(1,4,2:4)
imagesc(log10(A),om,log10(abs(H)))
axis xy
xlabel('Vibration amplitude [log]')
grid on
hold on
plot(log10([A(1) A(end)]),sqrt(alpha)*[1 1],'--k')
ylim([0 64])

subplot(141)
plot(10*log10(Pyy(:,end)),frec)
grid on
ylim([0 64])
set(gca,'XDir','reverse')
ylabel('Frequency [rad/s]')
xlabel('Power [dB]')