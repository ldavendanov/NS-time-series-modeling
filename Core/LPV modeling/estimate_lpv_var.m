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

%-- Building the representation basis according to the basis type
switch options.basis.type
    
    %-- Fourier basis
    case 'fourier'
        
        % The basis order must be odd when using the Fourier basis to
        % ensure numerical stability
        if mod(pa,2) == 0
            pa = pa-1;
            warning('Basis order must be odd when using the Fourier basis. Basis order set to p-1')
        else
            g = ones(pa,N);
            for j=1:(pa-1)/2
                g(2*j,:) = sin(j*2*pi*xi);
                g(2*j+1,:) = cos(j*2*pi*xi);
            end
        end
        
    %-- Hermite polynomials
    case 'hermite'
        g = ones(pa,N);
        g(2,:) = 2*xi;
        for j=3:pa
            g(j,:) = 2*xi.*g(j-1,:) - 2*(j-1)*g(j-2,:);
        end
        
end

%-- Selecting the indices of the basis to be used in the analysis
if isfield(options.basis,'indices')
    g = g(options.basis.indices,:);
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
        M.a = reshape(M.ParameterVector,n,n,pa,na);
        M.estimator = 'Ordinary Least Squares';
        
    case 'map_normal'
        Theta0 = options.estimator.Theta0;                                  % Prior mean parameter vector
        SigmaTh = options.estimator.SigmaTheta;                             % Prior parameter covariance
        sigmaW2 = options.estimator.sigmaW2;                                % Innovations variance
        
        M = mapNormal(Phi,y(tau),Theta0,SigmaTh,sigmaW2);
        M.a = reshape(M.ParameterVector,n,n,pa,na);
        M.estimator = 'Maximum A Posteriori - Normal Prior';
end

%-- Other fields in the model data structure
M.structure.na = na;
M.structure.pa = length(options.basis.indices);
M.structure.basis = options.basis;
M.model_type = 'LPV-AR';

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