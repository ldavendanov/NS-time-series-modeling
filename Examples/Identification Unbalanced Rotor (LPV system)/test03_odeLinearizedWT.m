clear
close all
clc

% Define WT rotor properties
System = WTrotorProperties;                                                 % Physical properties of the WT
n = 5;

% Simulation properties
fs0 = 40;                                                                   % Sampling rate (Hz)
T = 10*60;                                                                  % Sampling period (s)
N = (T*fs0)+1;                                                              % Number of samples
t = (0:N-1)/fs0;                                                            % Time vector

Omega = 2*pi*linspace(0,20)/60;                                             % Rotor speed

% Create excitation
TauM = SimulateExcitation(N,fs0);                                           % Excitation time series - Torque at each blade (N.m)

Nf = 2^12;
Pyy = zeros(Nf/2+1,n,numel(Omega));

parfor k=1:numel(Omega)

    % Integrate EOM
    [~,z] = ode45( @(t,z)WTRotorLinEOM( t,z, System, Omega(k), TauM, fs0 ), t, zeros(10,1) );

    % Calculate acceleration response
    y = zeros(N,n);
    for i=1:N
        dz = WTRotorLinEOM( i/fs0, z(i,:)', System, Omega(k), TauM, fs0 );
        y(i,:) = dz(6:10);
    end

    Pyy(:,:,k) = pwelch( y, hann(Nf), Nf*3/4, Nf, fs0 );

end

%%
close all
clc

freq = fs0*(0:Nf/2)/Nf;

figure
tiledlayout(1,n)

for i=1:n
    nexttile
    imagesc(60*Omega/(2*pi),freq, log10( squeeze(Pyy(:,i,:)) ))
    axis xy
    ylim([0 4])
end



%%

% %% Plot results
% close all
% clc
% 
% figure
% tiledlayout(5,1)
% 
% for i=1:5
%     nexttile
%     plot(t/60,y(:,i))
% end
% 
% Nf = 2^14;
% figure
% pwelch(y,hann(Nf),Nf*3/4,Nf,fs0)
% xlim([0 5])
% hold on
% for i=1:3
%     xline( i*OmegaRef/(2*pi) )
% end
% 
% %% Frozen dynamic analysis
% close all
% clc
% 
% Tspan = linspace(0,2*pi/OmegaRef);
% 
% fn = zeros(numel(Tspan),10);
% zeta = zeros(numel(Tspan),10);
% 
% for i=1:numel(Tspan)
%     [M,C,K] = WTSystemMatrices( System, OmegaRef, Tspan(i) );
%     A = [zeros(5) eye(5);
%          -M\[K C]];
%     lambda = eig(A);
%     fn(i,:) = abs(lambda)/(2*pi);
%     zeta(i,:) = -cos(angle(lambda));
% 
%     [fn(i,:),ind] = sort(fn(i,:),'ascend');
%     zeta(i,:) = zeta(i,ind);
% end
% 
% Nf = 2^14;
% 
% figure('Position',[100 100 900 600])
% tiledlayout(3,1)
% 
% nexttile
% pwelch(y,hann(Nf),Nf*3/4,Nf,fs0)
% xlim([0 3])
% hold on
% for i=1:3
%     xline( i*OmegaRef/(2*pi) )
% end
% 
% nexttile(2,[2 1])
% plot(fn,100*zeta)
% xlim([0 3])
% grid on
% xlabel('Frequency (Hz)')
% ylabel('Damping ratio (%)')
