function Se = tdvar(x,options)
% Function to compute the estimate of the time-dependent variance of the
% process 'x'
% Inputs : 
%   x       : Signal to compute the time-dependent variance
%   options : Options of the innovations variance estimation
%     options.Type : Type of estimator
%                    - 'c'   : Constant
%                    - 'mw'  : Moving window
%                    - 'iv'  : Instantaneous variance
%                    - 'cml' : Conditional maximum likelihood
%     options.Nw : Number of samples for the moving window variance estimator
%     options.ind : Vector with the basis indices for the 'iv' and 'cml' methods
%     options.basis.type : Type of the basis 
%                          - 'sinus'   -> Sinusoidal basis (default)
%                          - 'poly'    -> Polynomial basis (Chebyshev polynomials)
%                          - 'bspline' -> B-spline basis
%     options.basis.par : Basis parameters (for some types of basis)
%     options.MaxIter : Maximum number of iterations (scalar only for 'cml' method) 
%     options.TolFun  : Tolerance of the change of the objective function ('cml' method)
%     options.TolPar  : Tolerance of the change of the parameters ('cml' method)
%
% Outputs :
%  Se   : Estimate of the time-dependent variance
%   Se.iv, Se.cml : Projection parameters obtained by the respective estimation method
%   Se.time : Estimated time-dependent variance
%       
% Created by : David Avendano - April 2013 - Ver 1.0
%               All rights reserved
%--------------------------------------------------------------------------

% Checking the input and producing default values
if nargin < 2
    fprintf('Incomplete information to estimate the model\n')
    return
else
    % Extracting information from the input
    N = length(x);
    
    % Default variance estimator
    if ~isfield(options,'Type'), options.Type = 'c'; end
    
    % Assigning default values for the 'cml' estimation method
    if strcmp(options.Type,'cml')
        if ~isfield(options,'MaxIter'), 
            options.MaxIter = 1e4;
        end
        if ~isfield(options,'TolFun'), 
            options.TolFun = 1e-6;
        end
        if ~isfield(options,'TolPar'), 
            options.TolPar = 1e-6;
        end
    end
    
    % Estimate the innovations variance
    switch options.Type
        case 'c'    % Constant innovations variance
            Se.time = var(x)*ones(1,N);
            
        case 'mw'   % Moving window extimator
            
            % Checking input
            if ~isfield(options,'Nw'), 
                Nw = 100;  
            else
                Nw = options.Nw;
            end
            
            % Computing the TD-variance
            Se.time = zeros(1,N);
            for t = Nw+1:N-Nw
                Se.time(t) = mean(x(t-Nw:t+Nw).^2);
            end
            
        case 'iv'   % Instantaneous variance estimator
            
            % Checking input
            if ~isfield(options,'ind')
                fprintf('Incomplete information to estimate the time dependent variance\n')
                return
            end
            
            % Default basis function
            if ~isfield(options,'basis'), options.basis = 'sinus'; end
            
            % Computing the basis
            Gbs = basis(N,options.ind,options.basis);
            
            % Computing the projection parameters
            Se.iv = wls(Gbs,abs(x)');
            Se.time = (Se.iv'*Gbs).^2;
            
        case 'cml'  % Conditional maximum likelihood method
            
            % Checking input
            if ~isfield(options,'ind')
                fprintf('Incomplete information to estimate the time dependent variance\n')
                return
            end
            
            % Default basis function
            if ~isfield(options,'basis'), options.basis = 'sinus'; end
            
            % Computing the basis
            Gbs = basis(N,options.ind,options.basis);
            
            % Computing the initial values with the IV method
            optiv = options;
            optiv.Type = 'iv';
            Se = tdvar(x,optiv);
            
            % Optimizing the values of the projection parameters based on
            % the CML method
            nl_options = optimset('fminsearch');
            nl_options.Display = 'iter';
            nl_options.MaxFunEvals = 10*options.MaxIter;
            nl_options.MaxIter = options.MaxIter;
            nl_options.TolFun = options.TolFun;
            nl_options.TolX = options.TolPar;
            
            Se.cml = fminsearch( @(se) cml( x, se, Gbs ), Se.iv, nl_options );
            Se.time = Gbs'*Se.cml;
            
    end
    
end

%--------------------------------------------------------------------------
function L = cml(e,se,Gbs)

se_time = Gbs'*se;
L = LogLikelihood(e,se_time);