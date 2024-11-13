function M = estimate_lpv_var(signals,structure,options)

%% Part 0 : Unpacking and checking input

%-- Unpacking the input
y = signals.response;                                                       % Response signal
xi = signals.scheduling_variables;                                          % Scheduling variable
[n,N] = size(y);                                                            % Signal length

%-- Model structure
na = structure.na;                                                              % AR order
pa = structure.pa;                                                              % Basis order

%-- Completing the options structure
options = check_input(signals,structure,options);

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
        M.a.time = zeros(n,n*na,N);
        for i=1:N
            M.a.time(:,:,i) = M.Parameters.Theta*kron(eye(n*na),g(:,i));
        end
        M.estimator = 'Ordinary Least Squares';

    case 'svd_ols'
        M = svd_ols(Phi,y(:,tau));
        M.a.projection = reshape(M.Parameters.Theta,n,n,pa,na);
        M.a.time = zeros(n,n*na,N);
        for i=1:N
            M.a.time(:,:,i) = M.Parameters.Theta*kron(eye(n*na),g(:,i));
        end
        M.estimator = 'Ordinary Least Squares - SVD algorithm';
        
    case 'map_normal'
        Theta0 = options.estimator.Theta0;                                  % Prior mean parameter vector
        Lambda0 = options.estimator.Lambda0;                                % Prior parameter covariance
        V0 = options.estimator.V0;                                          % Innovations variance
        nu = n - 1;

        M = mapNormalBatch(Phi,y(:,tau),Theta0,Lambda0,V0,nu);
        M.a.projection = reshape(M.Parameters.Theta,n,n,pa,na);
        M.a.time = zeros(n,n*na,N);
        for i=1:N
            M.a.time(:,:,i) = M.Parameters.Theta*kron(eye(n*na),g(:,i));
        end
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

%-- Other fields in the model data structure
M.structure.na = na;
M.structure.pa = length(options.basis.indices);
M.structure.basis = options.basis;
M.model_type = 'LPV-VAR';


%% ------------------------------------------------------------------------
function options = check_input(signals,structure,options)

[n,N] = size(signals.response);
na = structure.na;
pa = structure.pa;

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
    if ~isfield(options.estimator,'Lambda0')
        options.estimator.Lambda0 = 1e0*eye(n*na*pa);
    end
    if ~isfield(options.estimator,'V0')
        options.estimator.V0 = diag(0.1*var(signals.response,[],2));
    end
end