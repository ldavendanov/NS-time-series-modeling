function [yhat,criteria] = simulate_lpv_arx(signals,M)

%-- Unpacking the input
y = signals.response(:)';                                                   % Response signal
x = signals.excitation(:)';                                                 % Excitation signal
xi = signals.scheduling_variables(:)';                                      % Scheduling variable
[~,N] = size(y);                                                            % Signal length

Theta = M.ParameterVector;
sigmaW = M.InnovationsVariance;

na = M.structure.na;
nb = M.structure.nb;
pa = M.structure.pa;

%-- Constructing the representation basis
switch M.structure.basis.type
    case 'fourier'
        g = ones(pa,N);
        for j=1:(pa-1)/2
            g(2*j,:) = sin(j*xi(1,:));
            g(2*j+1,:) = cos(j*xi(1,:));
        end
    case 'hermite'
        g = ones(pa,N);
        g(2,:) = 2*xi;
        for j=3:pa
            g(j,:) = 2*xi.*g(j-1,:) - 2*(j-1)*g(j-2,:);
        end
        
end

if isfield(M.structure.basis,'indices')
    g = g(M.structure.basis.indices,:);
    pa = sum(M.structure.basis.indices);
end

%-- Constructing the lifted signal
Y = zeros(pa,N);
X = zeros(pa,N);
for j=1:pa
    X(j,:) =  x.*g(j,:);
    Y(j,:) = -y.*g(j,:);
end

%-- Constructing the regression matrix
PhiX = zeros((nb+1)*pa,N-max(na,nb));
PhiY = zeros(na*pa,N-max(na,nb));
tau = max(na,nb)+1:N;
for i=1:na
    PhiY((1:pa)+(i-1)*pa,:) = Y(:,tau-i);
end
for i=0:nb
    PhiX((1:pa)+i*pa,:) = X(:,tau-i);
end
Phi = [PhiY; PhiX];

%-- Calculating the prediction error
yhat = [zeros(1,na), Theta*Phi];
err = y(:,tau) - yhat(:,tau);

%-- Performance criteria
criteria.rss = sum(err.^2);
criteria.rss_sss = criteria.rss/sum(y.^2);
criteria.lnL = -(1/2)*( sum(log(2*pi*sigmaW) + err.^2/sigmaW) );