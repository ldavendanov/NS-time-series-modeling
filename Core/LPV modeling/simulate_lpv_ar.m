function [yhat,performance] = simulate_lpv_ar(signals,M)

%% Part 0 : Unpacking the input
y = signals.response(:)';
xi = signals.scheduling_variables(:)';
Theta = M.Parameters.Theta;

na = M.Structure.na;
pa = M.Structure.pa;
ps = M.Structure.ps;
[~,N] = size(y);

%% Part 1 : Constructing the representation basis

%-- Representation basis for the AR parameters
if isfield(M.Structure.basis,'ind_ba')
    ba = M.Structure.basis.ind_ba;
    Gba = lpv_basis(xi,ba,M.Structure.basis);
else
    ba = 1:pa;
    Gba = lpv_basis(xi,ba,M.Structure.basis);
end

%-- Representation basis for the innovations variance
if ps > 1
    if isfield(M.Structure.basis,'ind_bs')
        bs = M.Structure.basis.ind_bs;
        Gbs = lpv_basis(xi,bs,M.Structure.basis);
    else
        bs = 1:ps;
        Gbs = lpv_basis(xi,bs,M.Structure.basis);
    end
end

%% Part 2 : Constructing the regression matrix

%-- Constructing the lifted signal
Y = zeros(numel(ba),N);
for j=1:numel(ba)
    Y(j,:) = -y.*Gba(j,:);
end

%-- Constructing the regression matrix
Phi = zeros(na*numel(ba),N-na);
tau = na+1:N;
for i=1:na
    Phi((1:numel(ba))+(i-1)*numel(ba),:) = Y(:,tau-i);
end

%% Part 3 : Calculating predicitons and performance

%-- Calculating the prediction error
yhat = [zeros(1,na), Theta*Phi];
err = y(:,tau) - yhat(:,tau);
if M.Structure.ps > 1
    sigmaW2 = M.InnovationsVariance.S.Parameters.Theta*Gbs;
else
    sigmaW2 = M.InnovationsVariance.sigmaW2*ones(1,N);
end

%-- Performance criteria
performance.rss = sum(err.^2);
performance.rss_sss = performance.rss/sum(y.^2);
performance.lnL = -(1/2)*( sum(log(2*pi*sigmaW2(tau)) + err.^2./sigmaW2(tau)) );