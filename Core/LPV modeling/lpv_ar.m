function [A,sigmaW,SigmaTh,criteria] = lpv_ar(y,xi,order,options)

na = order(1);
pa = order(2);
[~,N] = size(y);

%-- Constructing the representation basis
switch options.basis.type
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

if isfield(options.basis,'indices')
    g = g(options.basis.indices,:);
    pa = sum(options.basis.indices);
end

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

%-- Computing the OLS parameter estimates
A = y(:,tau)/Phi;
yhat = [zeros(1,na), A*Phi];
err = y(:,tau) - yhat(:,tau);
sigmaW = var(err);

A = reshape(A,pa,na);

SigmaTh = sigmaW*eye(na*pa)/(Phi*Phi');

%-- Performance criteria
criteria.rss = sum(err.^2);
criteria.rss_sss = criteria.rss/sum(y.^2);
criteria.lnL = -(1/2)*( sum(log(2*pi*sigmaW) + err.^2/sigmaW) );
criteria.bic = log(N)*na*pa - 2*criteria.lnL;