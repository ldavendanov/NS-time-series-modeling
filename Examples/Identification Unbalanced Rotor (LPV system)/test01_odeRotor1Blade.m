clear
close all
clc

Jo = 1e2;
m = [1e3 180];
l = 2;
k = [20e4 40e4 20e4];
C = [5e2 1e3];

fs = 20;
OmegaRef = 2*pi*10/60;
T = 300;
N = (T*fs)+1;
t = (0:N-1)/fs;
[b,a] = butter(4,0.2);
TauM = 10 + 0.1*filter(b,a,randn(1,N));

[t,z] = ode45( @(t,z)RotorEOM( t,z,m,Jo,k,l,TauM, fs, C, OmegaRef  ), t, zeros(9,1) );
y = zeros(N,3);
for i=1:N
    dz = RotorEOM( i/fs, z(i,:)', m,Jo, k, l, TauM, fs, C, OmegaRef );
    y(i,:) = dz(5:7);
end

%%
close all
clc

figure
plot(t,y)
grid on
xlim([60 T])

figure
plot(t,z(:,8))

figure
plot(t,z(:,9))
hold on
plot(t,TauM)

%% Frequency analysis
clc
close all

Nf = 2^10;
figure
pwelch(y,hann(Nf),3*Nf/4,Nf,fs)


Nf = 2^10;
[Syy,ff,tt] = spectrogram( y(:,1), gausswin(Nf,8), Nf-4, Nf, fs );

figure
imagesc(tt,ff,log10(abs(Syy)))
axis xy


%% ------------------------------------------------------------------------
function dz = RotorEOM( t, z, m, Jo, k, l, TauM, fs, C, OmegaRef )

ind = max(1,round(t*fs));
q = z(1:4);
dq = z(5:8);
Mo = z(9);
alpha = q(3)+q(4);
alpha_dot = dq(3)+dq(4);

f_int = [-k(1)*q(1) + m(2)*l*alpha_dot.^2*cos(alpha)
         -k(2)*q(2) + m(2)*l*alpha_dot.^2*sin(alpha)
         -k(3)*q(3)
          Mo];
f_ext = [0 0 TauM(ind) 0]';
M = MassMatrix( alpha, m, l, Jo );


dz = [ dq;
       M\(f_int+f_ext)
       0];
dz(end) = C(1)*(OmegaRef-dq(4)) - C(2)*dz(8);

end

function M = MassMatrix( alpha, m, l, Jo )

M = [            sum(m)                 0 -m(2)*l*sin(alpha) -m(2)*l*sin(alpha)
                      0            sum(m)  m(2)*l*cos(alpha)  m(2)*l*cos(alpha)
     -m(2)*l*sin(alpha) m(2)*l*cos(alpha)           m(2)*l^2          m(2)*l^2
     -m(2)*l*sin(alpha) m(2)*l*cos(alpha)           m(2)*l^2     Jo + m(2)*l^2];

end