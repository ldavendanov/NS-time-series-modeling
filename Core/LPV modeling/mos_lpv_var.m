function Performance = mos_lpv_var(signals,order,options)
%% Model order selection for LPV-VAR model

%% Part 0 : Unpacking and checking input

%-- Unpacking the input
y = signals.response;                                                       % Response signal
xi = signals.scheduling_variables;                                          % Scheduling variable
[n,N] = size(y);                                                            % Signal length

%-- Model structure
na = order(1);                                                              % AR order
pa = order(2);                                                              % Basis order

%-- Completing the options structure
options = check_input(signals,order,options);

%% Part 1 : Constructing the representation basis

%-- Construct the parameter projection basis
g = lpv_basis(xi,1:pa,options.basis);

%-- Selecting the indices of the basis to be used in the analysis
if isfield(options.basis,'indices')
    pa = sum(options.basis.indices);
end

%% Part 2 : Building the regression matrix

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

%% Part 3 : Calculating the LPV-VAR models with increasing model order - QR algorithm

[Q,R] = qr(Phi','econ');
Y = y(:,tau);
Y0 = (Y/Q');

rss_sss = zeros(na,1);
bic = zeros(na,1);
spp = zeros(na,1);
CN = zeros(na,1);
for i = 1:na

    fprintf('Computing for na = %3d and pa = %3d\n',i,0)

    for k=1:pa
        
        fprintf('\b\b\b\b%3d\n',k)

        indx = false(1,n*pa*na);
        for j=1:i
            indx( (1:k*n) + (j-1)*pa*n ) = true;
        end

        r = R(:,indx);
        Theta = Y0/r';
        err = Y - Theta*Phi(indx,:);
        SigmaW = cov(err');

        %-- Performance criteria
        rss = sum(sum(err.^2));                                                     % Residual Sum of Squares ( RSS )
        rss_sss(i,k) = rss/sum(sum(Y.^2));                                          % Residual Sum of Squares over Series Sum of Squares
        lnL = -(1/2)*( trace( log(2*pi*det(SigmaW)) + (err*err')/SigmaW ) );
        bic(i,k) = log(numel(Y))*numel(Theta) + numel(Y)*log(det(SigmaW));          % Bayesian Information Criterion ( BIC )
        spp(i,k) = N*n/numel(Theta);                                                % Samples Per Parameter ( SPP )
        CN(i,k) = cond(r);                                                          % Condition Number of the inverted matrices - Numerical accurady of the estimate
    end
end

Performance.rss_sss = rss_sss;
Performance.bic = bic;
Performance.spp = spp;
Performance.CN = CN;

%% ------------------------------------------------------------------------
function options = check_input(signals,order,options)

na = order(1);
pa = order(2);

if ~isfield(options,'basis')
    options.basis.type = 'hermite';                                         % Default basis type : Hermite polynomials
    options.basis.indices = true(1,pa);
end

if ~isfield(options.basis,'indices')
    options.basis.indices = true(1,pa);
end

if ~isfield(options,'estimator')
    options.estimator.type = 'ols';                                         % Default estimator: Ordinary Least Squares
end

if strcmp(options.estimator.type,'map_normal')
    if ~isfield(options.estimator,'Theta0')
        options.estimator.Theta0 = zeros(na*pa,1);
    end
    if ~isfield(options.estimator,'SigmaTheta')
        options.estimator.SigmaTheta = eye(na*pa);
    end
    if ~isfield(options.estimator,'sigmaW2')
        options.estimator.sigmaW2 = 1;
    end
end