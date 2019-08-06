function [A,Se,e,criteria] = fstar(y,na,ind,options)
%--------------------------------------------------------------------------
% Function to estimate a FS-TAR model of the form
%       y[t] = -sum_{i=1}^{na} a_i[t] * y[t-i] + w[t]
%       a_i[t] = sum_{k=1}^{pa} a_{i,k} * Gba_k[t]
%
% Inputs :  
%  y       -> Signal to be modeled (1xN vector)
%  na      -> AR order (integer scalar)
%  ind     -> Indices of the functional basis (integer vector)
%   ind.ba : Indices of the AR functional basis
%   ind.bs : Indices of the innovations variance functional basis (ignored in semi-parametric models) 
%  options -> General options of the model
%   options.basis : Structure with the model basis description
%     options.basis.type : Type of the basis 
%                          - 'sinus'   -> Sinusoidal basis (default)
%                          - 'poly'    -> Polynomial basis (Chebyshev polynomials)
%                          - 'bspline' -> B-spline basis
%     options.basis.par : Basis parameters (for some types of basis)
%   options.ParamEstim : Options of the projection parameter estimation
%     options.ParamEstim.Type : Type of estimator
%                                - 'ols' ordinary least squares,
%                                - 'wls' weighted least squares,
%                                - 'ml' maximum likelihood,
%                                - 'ms' multi stage
%     options.ParamEstim.weights : Weights of the wls estimator (1xN vector)
%     options.ParamEstim.MaxIter : Maximum number of iterations (scalar only for 'ml' and 'ms' estimators) 
%     options.ParamEstim.TolFun  : Tolerance of the change of the objective function
%     options.ParamEstim.TolPar  : Tolerance of the change of the parameters
%   options.VarEstim : Options of the innovations variance estimation
%     options.VarEstim.Type : Type of estimator
%                             - 'c'   : Constant
%                             - 'mw'  : Moving window
%                             - 'iv'  : Instantaneous variance
%                             - 'cml' : Conditional maximum likelihood
%     options.VarEstim.Nw : Number of samples for the moving window variance estimator
%     options.VarEstim.MaxIter : Maximum number of iterations (scalar only for 'ml' and 'ms' estimators) 
%     options.VarEstim.TolFun  : Tolerance of the change of the objective function
%     options.VarEstim.TolPar  : Tolerance of the change of the parameters
%
% Outputs : 
%  A    -> Structure with the parameters of the model
%           A.ols, A.wls, A.ml, A.ms : Estimated projection parameters
%           obtained by the respective estimator class
%           A.time : TAR parameters (projected in time)
%  S    -> Structure with the estimated innovations variance
%           S.iv, S.cml : Estimated innovations variance projection
%           parameters with the respective estimator class
%           S.time : Innovations variance (projected in time)
%  e    -> Estimation error
%  criteria -> Estimation and model performance criteria
%       
% Created by : David Avendano - April 2013 - Ver 1.0
%               All rights reserved
%--------------------------------------------------------------------------

% Checking the input and producing default values
if nargin < 4
    fprintf('Incomplete information to estimate the model\n')
    return
else
    % Extracting information from the input
    N = length(y);
    
    % Default basis function
    if ~isfield(options,'basis'), options.basis = 'sinus'; end
    
    % Default projection parameter estimator
    if ~isfield(options,'ParamEstim'), options.ParamEstim.Type = 'ols'; end
    
    % Default variance estimator
    if ~isfield(options,'VarEstim'), options.VarEstim.Type = 'c'; end
    
    % Assigning default values for the 'ml' and 'ms' parameter estimation methods
    if strcmp(options.ParamEstim.Type,'ml')
        if ~isfield(options.ParamEstim,'MaxIter'), 
            options.ParamEstim.MaxIter = 1e4;
        end
        if ~isfield(options.ParamEstim,'TolFun'), 
            options.ParamEstim.TolFun = 1e-6;
        end
        if ~isfield(options.ParamEstim,'TolPar'), 
            options.ParamEstim.TolPar = 1e-6;
        end
    end
    
    if strcmp(options.ParamEstim.Type,'ms')
        if ~isfield(options.ParamEstim,'MaxIter'), 
            options.ParamEstim.MaxIter = 100;
        end
        if ~isfield(options.ParamEstim,'TolFun'), 
            options.ParamEstim.TolFun = 1e-8;
        end
        if ~isfield(options.ParamEstim,'TolPar'), 
            options.ParamEstim.TolPar = 1e-8;
        end
    end
       
    % Assigning default values for the 'cml' innovations variance estimation methods
    if strcmp(options.VarEstim.Type,'cml')
        if ~isfield(options.VarEstim,'MaxIter'), 
            options.VarEstim.MaxIter = 1e3;
        end
        if ~isfield(options.VarEstim,'TolFun'), 
            options.VarEstim.TolFun = 1e-4;
        end
        if ~isfield(options.VarEstim,'TolPar'), 
            options.VarEstim.TolPar = 1e-4;
        end
    end
    
    % Computing the basis and the regression matrix
    Gba = basis(N,ind.ba,options.basis);            % Basis
    Phi = RegMatrix(y,Gba,na,ind.ba,N);             % Regression matrix
    
    % Going to the FS-TAR estimator
    [A,Se,e,criteria] = estimate(y,na,ind,options,Phi,Gba);
end

%--------------------------------------------------------------------------
function [A,Se,e,criteria] = estimate(y,na,ind,options,Phi,Gba)

p = numel(ind.ba);

%--- Computing the initial estimate (OLS/WLS) -----------------------------
if strcmp( options.ParamEstim.Type, 'ols' ) || strcmp( options.ParamEstim.Type, 'ms' ) || ...
    strcmp( options.ParamEstim.Type, 'ml' )
    
    % Estimate the projection parameters
    [A.ols,e,criteria] = wls(Phi,y(na+1:end)');                             % OLS estimate of the parameters
    A.time = reshape(A.ols,p,na)'*Gba;                                      % FS-TAR parameters
    
    % Estimate the innovations variance
    options.VarEstim.ind = ind.bs;
    options.VarEstim.basis = options.basis;
    Se = tdvar(e,options.VarEstim);                                         % Innovations variance
    
elseif strcmp( options.ParamEstim.Type, 'wls' )
    
    % Estimate the projection parameters
    [A.wls,e,criteria] = wls(Phi,y(na+1:end)',options.ParamEstim.weights);  % WLS estimate of the parameters
    A.time = reshape(A.wls,p,na)'*Gba;                                      % FS-TAR parameters
    
    % Estimate the innovations variance
    options.VarEstim.ind = ind.bs;
    options.VarEstim.basis = options.basis;
    Se = tdvar(e,options.VarEstim);                                         % Innovations variance

end

%--- Posterior refinement of the parameter estimates with the MS and ML methods 
switch options.ParamEstim.Type
    case 'ms'   % Multi Stage method
        % Initializing the criteria to break the iterations
        a0 = A.ols;                                                         % Previous value of the projection parameters
        mse0 = mean(e.^2);                                                  % Previous value of the Mean Squared Error
        w = Se.time;                                                        % Weights for the WLS method
        
        disp('Iteration - Step size - Func. Value - First diff.')
        for k = 1:options.ParamEstim.MaxIter
            
            % Step 1 : Compute the WLS projection parameter estimate
            [A.ms,e,criteria] = wls(Phi,y(na+1:end)',w);                    % WLS estimate of the parameters
            A.time = reshape(A.ms,p,na)'*Gba;                               % FS-TAR parameters
            
            % Step 2 : Update the time-dependent innovations variance
            Se = tdvar(e,options.VarEstim);
            w = sqrt(Se.time);
            
            % Checking the convergence of the method
            mse = mean(e.^2);
            a = A.ms;
            d_mse = abs(mse-mse0)/mse0;
            d_a = norm(a-a0)./norm(a0);
            mse0 = mse;
            a0 = a;
            
            fprintf('    %5d - %1.3e -   %1.3e -   %1.3e\n',k,d_a,mse,d_mse)
            
            if d_mse <= options.ParamEstim.TolFun
                disp('The iteration MSE is lower than the objective MSE. Optimization finished...')
                break
            end
            
            if d_a <= options.ParamEstim.TolPar
                disp('The change in the parameters is lower than the selected threshold. Optimization finished...')
                break
            end
           
            if k == options.ParamEstim.MaxIter
                disp('Maximum number of iterations reached.')
            end
            
        end
        
    case 'ml'   % Maximum likelihood method
        
        % Optimizing the values of the projection parameters based on the ML method
        nl_options = optimset('fminsearch');
        nl_options.Display = 'iter';
        nl_options.MaxFunEvals = 10*options.ParamEstim.MaxIter;
        nl_options.MaxIter = options.ParamEstim.MaxIter;
        nl_options.TolFun = options.ParamEstim.TolFun;
        nl_options.TolX = options.ParamEstim.TolPar;
        
        A.ml = fminsearch( @(a) ml( a, Phi, y(na+1:end)', options.VarEstim ), A.ols, nl_options );
        A.time = reshape(A.ml,p,na)'*Gba;                                   % FS-TAR parameters
        e = y(na+1:end)' - Phi'*A.ml;
        
        % Estimate the innovations variance
        options.VarEstim.ind = ind.bs;
        options.VarEstim.basis = options.basis;
        Se = tdvar(e,options.VarEstim);                                     % Innovations variance
        
end

% Computing the criteria for the modeling performance
N = length(e);
criteria.rss = sum(e.^2);
criteria.rss_sss = criteria.rss/sum(y.^2);
criteria.mse = mean(e.^2);
criteria.logL = loglikelihood(e,Se.time);
criteria.bic = - criteria.logL + log(N)*(na*p)/2;
criteria.spp = (na*p)/N;
criteria.CondPhi = cond(Phi*Phi');

%--------------------------------------------------------------------------
function Phi = RegMatrix(y,Gba,na,ind,N)
% Computes the regression matrix of the FS-TAR model based on the output
% and the basis

p = numel(ind);
Phi = zeros(na*p,N-na);

for i=1:na
    Phi((i-1)*p+1:i*p,:) = Gba(:,na+1:N).*repmat(y(na+1-i:N-i),p,1);
end

%--------------------------------------------------------------------------
function L = ml( a, Phi, Y, options)

e = Y - Phi'*a;
Se = tdvar(e,options);
L = -loglikelihood(e,Se.time);
