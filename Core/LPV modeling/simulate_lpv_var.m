function [yhat,performance] = simulate_lpv_var(signals,M)

%% Part 0 : Unpacking the input
y = signals.response;
xi = signals.scheduling_variables;
Theta = M.Parameters.Theta;

na = M.structure.na;
pa = M.structure.pa;
[n,N] = size(y);                                                            % Signal length

%% Part 1 : Constructing the representation basis

ind = 1:pa;
g = lpv_basis(xi,ind(M.structure.basis.indices),M.structure.basis);
pa = sum(M.structure.basis.indices);

%% Part 2 : Constructing the regression matrix

%-- Constructing the lifted signal
Y = zeros(n*pa,N);
for j=1:pa
    Y((1:n)+n*(j-1),:) = -y.*repmat(g(j,:),n,1);
end

%-- Constructing the regression matrix
Phi = zeros(n*na*pa,N-na);
tau = na+1:N;
for i=1:na
    Phi((1:n*pa)+(i-1)*n*pa,:) = Y(:,tau-i);
end

%% Part 3 : Calculating predicitons and performance

%-- Calculating the prediction error
yhat = [zeros(n,na), Theta*Phi];
err = y(:,tau) - yhat(:,tau);
SigmaW = M.InnovationsCovariance.SigmaW;

%-- Performance criteria
performance.rss = sum(sum(err.^2));
performance.rss_sss = performance.rss/sum(sum(y.^2));
% performance.lnL = -(1/2)*( trace( log(2*pi*det(SigmaW)) + (err'/SigmaW)*err ) );