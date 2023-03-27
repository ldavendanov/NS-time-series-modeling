function M = estimate_lpv_var(signals,order,options)

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

%% Part 3 : Calculating the estimate

switch options.estimator.type
    %-- Ordinary Least Squares estimator
    case 'ols'
        M = ols(Phi,y(:,tau));
        M.a.projection = reshape(M.Parameters.Theta,n,n,pa,na);
        M.a.time = M.Parameters.Theta*g;
        M.estimator = 'Ordinary Least Squares';

    case 'svd_ols'
        M = svd_ols(Phi,y(:,tau));
        M.a.projection = reshape(M.Parameters.Theta,n,n,pa,na);
        M.estimator = 'Ordinary Least Squares - SVD algorithm';
        
    case 'map_normal'
        Theta0 = options.estimator.Theta0;                                  % Prior mean parameter vector
        SigmaTh = options.estimator.SigmaTheta;                             % Prior parameter covariance
        sigmaW2 = options.estimator.sigmaW2;                                % Innovations variance
        nu = 1;

        M = mapNormalBatch(Phi,y(:,tau),Theta0,SigmaTh,sigmaW2,nu);
        M.a.projection = reshape(M.Parameters.Theta,n,n,pa,na);
        M.estimator = 'Maximum A Posteriori - Normal Prior';
end

%-- Project parameters and parameter variance on time
M.a.time = zeros(n,n,na,N);
for i=1:N
    M.a.time(:,:,:,i) = squeeze(g(1,i)*M.a.projection(:,:,1,:));
    for j=2:pa
        M.a.time(:,:,:,i) = M.a.time(:,:,:,i) + g(j,i)*squeeze(M.a.projection(:,:,j,:));
    end
end

% M.a.time = zeros(n,n,N,na);
% for k=1:N
%     for i=1:na
%         for j=1:pa
%             M.a.time(:,:,k,i) = M.a.time(:,:,k,i) + M.a.projection(:,:,j,i).*g(j,k);
%         end
%     end
% end

%-- Other fields in the model data structure
M.structure.na = na;
M.structure.pa = length(options.basis.indices);
M.structure.basis = options.basis;
M.model_type = 'LPV-AR';

%% ------------------------------------------------------------------------
function options = check_input(signals,order,options)

[n,N] = size(signals.response);
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
        options.estimator.Theta0 = zeros(n,n*na*pa);
    end
    if ~isfield(options.estimator,'SigmaTheta')
        options.estimator.SigmaTheta = 1e0*eye(na*pa);
    end
    if ~isfield(options.estimator,'sigmaW2')
        options.estimator.sigmaW2 = diag(0.1*var(signals.response,[],2));
    end
end