function [yhat,performance] = simulate_lpv_ar(signals,M)

%% Part 0 : Unpacking the input
y = signals.response(:)';
xi = signals.scheduling_variables(:)';
Theta = M.ParameterVector;

na = M.structure.na;
pa = M.structure.pa;
[~,N] = size(y);

%% Part 1 : Constructing the representation basis
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

%-- Selecting the indices of the basis to be used in the analysis
g = g(M.structure.basis.indices,:);
pa = sum(M.structure.basis.indices);

%% Part 2 : Constructing the regression matrix

%-- Constructing the lifted signal
Y = zeros(pa,N);
for j=1:pa
    Y(j,:) = -y.*g(j,:);
end

%-- Constructing the regression matrix
Phi = zeros(na*pa,N-na);
tau = na+1:N;
for i=1:na
    Phi((1:pa)+(i-1)*pa,:) = Y(:,tau-i);
end

%% Part 3 : Calculating predicitons and performance

%-- Calculating the prediction error
yhat = [zeros(1,na), Theta*Phi];
err = y(:,tau) - yhat(:,tau);
sigmaW2 = M.InnovationsVariance;

%-- Performance criteria
performance.rss = sum(err.^2);
performance.rss_sss = performance.rss/sum(y.^2);
performance.lnL = -(1/2)*( sum(log(2*pi*sigmaW2) + err.^2/sigmaW2) );