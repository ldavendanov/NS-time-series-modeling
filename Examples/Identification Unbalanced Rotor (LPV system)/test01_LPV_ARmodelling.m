clear
close all
clc

fs0 = 100;
fs = 25;
T = 600;
N = T*fs0 + 1;
t = (0:N-1)/fs0;
OmegaRef = 2*pi*10/60;
[b,a] = butter( 6,  0.1 );
Mo = 10 + 2*filter(b,a,randn(1,N));

figure
pwelch(Mo)

[t,z] = ode45( @(t,z)UnbalancedRotorODE4dof( t, z, OmegaRef, Mo, fs0 ), t, zeros(9,1) );

y = zeros(N,3);
for i=1:N
    dz = UnbalancedRotorODE4dof(i/fs0,z(i,:)',OmegaRef,Mo,fs0);
    y(i,:) = dz(1:3);
end

beta = mod( z(:,4), 2*pi )/(2*pi);

[b,a] = butter(4,2*0.5/fs,"high");
y = resample(y(:,2),fs,fs0);
t = t(1:4:end);
beta = beta(1:4:end);
N = numel(t);

s = [cos(2*pi*beta) sin(2*pi*beta)];

Ys = (y(3001:end)'*s(3001:end,:))/(N-3000);

y = y - 2*s*Ys';

%% Plot time series
close all
clc

figure
plot(t,y,t,s*Ys')

%% Plot results
close all
clc

Nf = 2^12;
figure
pwelch([y s],hann(Nf),3*Nf/4,Nf,fs)

Nf = 2^10;
figure
% tiledlayout(3,1)

for i=1
    [Syy,ff,tt] = spectrogram( y(:,i), gausswin(Nf,8), Nf-4, Nf, fs );

    nexttile
    imagesc(tt,ff,10*log10(abs(Syy)))
    axis xy
end

%% Estimate LPV-AR model
close all
clc

addpath(genpath('..\..\Core'))

ind_train = 5e3+(1:5e3);
signals.response = y(ind_train,1)';                                           % Response ( edgewise acceleration )
signals.scheduling_variables = beta(ind_train)';                              % Scheduling variable ( rotor azimuth )

na_max = 10;
pa_max = 25;

na = 1:na_max;
pa = 1:2:pa_max;

rss_sss = zeros(numel(na),numel(pa));
bic = zeros(numel(na),numel(pa));

fN = zeros(na_max*pa_max,numel(na),numel(pa));
zeta = zeros(na_max*pa_max,numel(na),numel(pa));

for i = 1:numel(na)
    for j=1:numel(pa)

        structure.na = na(i);
        structure.pa = pa(j);

        options.basis.type = 'fourier';                                             % Type of functional basis
        options.basis.ind_ba = 1:structure.pa;

        M = estimate_lpv_ar(signals,structure,options);
        rss_sss(i,j) = M.Performance.rss_sss;
        bic(i,j) = M.Performance.bic;

        A_lft = [M.Parameters.Theta;
                  eye((na(i)-1)*pa(j),na(i)*pa(j))];
        D = eig(A_lft);

        fN(1:na(i)*pa(j),i,j) = abs(log(D))*fs/(2*pi);
        zeta(1:na(i)*pa(j),i,j) = -cos(angle(log(D)));

    end
end

%% Analysis of model order selection results
close all
clc

na_opt = 4;
pa_opt = 9;

figure('Position',[100 100 900 600])
tiledlayout(2,2)

nexttile
surf(na,pa,(rss_sss'*100))
xlabel('Model order')
ylabel('Basis order')
zlabel('RSS/SSS (%)')
grid on
view(30,40)

nexttile
plot(pa,(rss_sss(na_opt,:)'*100),'-o')
xlabel('Basis order')
ylabel('RSS/SSS (%)')
grid on


nexttile
surf(na,pa,bic')
xlabel('Model order')
ylabel('Basis order')
zlabel('BIC')
grid on
view(30,40)

nexttile
plot(pa,bic(na_opt,:),'-o')
xlabel('Basis order')
ylabel('BIC')
grid on

%% Frequency stabilization diagrams
close all
clc

zeta_max = 0.1;
z_lvl = linspace(0,zeta_max);
clr = parula(numel(z_lvl)-1);

figure('Position',[100 100 600 600])
tiledlayout(3,1)

Nf = 2^12;
nexttile
pwelch(y,hann(Nf),3*Nf/4,Nf,fs)

nexttile(2,[2 1])
FN = squeeze(fN(:,:,(pa_opt-1)/2));  FN = FN(:);
Z = squeeze(zeta(:,:,(pa_opt-1)/2));  Z = Z(:);
NA = squeeze( repmat(na, na_max*pa_max, 1) ); NA = NA(:);

for i=1:numel(z_lvl)-1
    ind = Z > z_lvl(i) & Z <= z_lvl(i+1);
    plot(FN(ind),NA(ind),'.','Color',clr(i,:))
    hold on
end
grid on
xlim([0 fs/2]), ylim([0 na_max])
xlabel('Frequency (Hz)')
ylabel('AR order')
cbar = colorbar;
cbar.Ticks = (0:5)/5;
cbar.TickLabels = linspace(0,zeta_max*100,6);
cbar.Label.String = 'Damping ratio (%)';


figure('Position',[800 100 600 600])
tiledlayout(3,1)

Nf = 2^12;
nexttile
pwelch(y,hann(Nf),3*Nf/4,Nf,fs)

nexttile(2,[2 1])
FN = squeeze(fN(:,na_opt,:));  FN = FN(:);
Z = squeeze(zeta(:,na_opt,:));  Z = Z(:);
PA = squeeze( repmat(pa, na_max*pa_max, 1) ); PA = PA(:);

for i=1:numel(z_lvl)-1
    ind = Z > z_lvl(i) & Z <= z_lvl(i+1);
    plot(FN(ind),PA(ind),'.','Color',clr(i,:))
    hold on
end
grid on
xlim([0 fs/2]), ylim([0 pa_max])
ylabel('Basis order')
cbar = colorbar;
cbar.Ticks = (0:5)/5;
cbar.TickLabels = linspace(0,zeta_max*100,6);
cbar.Label.String = 'Damping ratio (%)';

%% Estimate LPV-AR model with optimal order
close all
clc

structure.na = na_opt;
structure.pa = pa_opt;

options.basis.type = 'fourier';                                             % Type of functional basis
options.basis.ind_ba = 1:structure.pa;

M = estimate_lpv_ar(signals,structure,options);

figure
plot(t(ind_train),M.ar_part.a_time)

figure
imagesc(M.ar_part.a(1:pa_opt,1:pa_opt))

%%
close all
clc

na = structure.na;
pa = structure.pa;
A_lft = [M.Parameters.Theta;
         eye((na-1)*pa,na*pa)];
[V,D] = eig(A_lft);

lambda = diag(D);

figure
plot(real(lambda),imag(lambda),'.')
zgrid
axis square
