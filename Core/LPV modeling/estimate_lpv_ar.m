function M = estimate_lpv_ar(signals,structure,options)

%% Part 0 : Unpacking and checking input

%-- Unpacking the input
y = signals.response(:)';                                                   % Response signal
xi = signals.scheduling_variables(:)';                                      % Scheduling variable
[~,N] = size(y);                                                            % Signal length

%-- Model structure
na = structure.na;                                                          % AR order
pa = structure.pa;                                                          % Basis order for AR coefficients
if isfield(structure,'ps')
    ps = structure.ps;                                                      % Basis order for innovations variance
else
    ps = 1;                                                                 % Single basis (constant)
end

%-- Completing the options structure
options = check_input(signals,structure,options);

%% Part 1 : Constructing the representation basis

%-- Representation basis for the AR parameters
if isfield(options.basis,'ind_ba')
    ba = options.basis.ind_ba;
    Gba = lpv_basis(xi,ba,options.basis);
else
    ba = 1:pa;
    Gba = lpv_basis(xi,ba,options.basis);
end

%-- Representation basis for the innovations variance
if ps > 1
    if isfield(options.basis,'ind_bs')
        bs = options.basis.ind_bs;
        Gbs = lpv_basis(xi,bs,options.basis);
    else
        bs = 1:ps;
        Gbs = lpv_basis(xi,bs,options.basis);
    end
else
    bs = 1;
end

%% Part 2 : Building the regression matrix

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

%% Part 3 : Calculating the estimate

switch options.estimator.type
    
    %-- Ordinary Least Squares estimator
    case 'ols'
        M = ols(Phi,y(tau));
        M.ar_part.a = reshape(M.Parameters.Theta,numel(ba),na);
        M.ar_part.a_time = M.ar_part.a'*Gba;
        M.Estimator = 'Ordinary Least Squares';
        
        %-- Instantaneous Innovations Variance estimate
        if ps > 1            
            M = InstantaneousVariance( y(tau), M, Phi, Gbs(:,tau) );
        end
        
    %-- Bayesian (Maximum a Posteriori) with normal prior
    case 'map_normal'
        Theta0 = options.estimator.Theta0;                                  % Prior mean parameter vector
        SigmaTh = options.estimator.SigmaTheta;                             % Prior parameter covariance
        sigmaW2 = options.estimator.sigmaW2;                                % Innovations variance
        
        M = mapNormal(Phi,y(tau),Theta0,SigmaTh,sigmaW2);
        M.ar_part.a = reshape(M.ParameterVector,numel(ba),na);
        M.ar_part.a_time = M.a'*Gba;
        M.Estimator = 'Maximum A Posteriori - Normal Prior';
        
    %-- Multi-Stage method
    case 'multi-stage'
        M = MultiStageML( Phi, y(tau), Gbs(:,tau) );   
        M.ar_part.a = reshape(M.Parameters.Theta,numel(ba),na);
        M.ar_part.a_time = M.ar_part.a'*Gba;
        M.InnovationsVariance.s = M.InnovationsVariance.S.Parameters.Theta;
        M.InnovationsVariance.sigmaW2 = M.InnovationsVariance.s*Gbs;
        
        M.Estimator = 'Multi-Stage ML';
end

%-- Other fields in the model data structure
M.Structure.na = na;
M.Structure.pa = pa;
M.Structure.ps = ps;
M.Structure.basis = options.basis;
M.Structure.basis.ind_ba = ba;
M.Structure.basis.ind_bs = bs;
M.model_type = 'LPV-AR';

%% ------------------------------------------------------------------------
function options = check_input(signals,structure,options)

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
        options.estimator.Theta0 = zeros(na*pa,1);
    end
    if ~isfield(options.estimator,'SigmaTheta')
        options.estimator.SigmaTheta = eye(na*pa);
    end
    if ~isfield(options.estimator,'sigmaW2')
        options.estimator.sigmaW2 = 1;
    end
end