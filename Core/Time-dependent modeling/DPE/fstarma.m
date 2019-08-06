function [A,C,Se,e,criteria] = fstarma(y,na,nc,ind,options)
%--------------------------------------------------------------------------
% Function to estimate a FS-TARMA model of the form
%       y[t] = - sum_{i=1}^{na} a_i[t] * y[t-i] + sum_{i=1}^{nc} c_i[t] * w[t-i] + w[t]
%       a_i[t] = sum_{k=1}^{pa} a_{i,k} * Gba_k[t]
%       c_i[t] = sum_{k=1}^{pc} c_{i,k} * Gbc_k[t]
%
% Inputs :  
%  y       -> Signal to be modeled (1xN vector)
%  na      -> AR order (integer scalar)
%  nc      -> MA order (integer scalar)
%  ind     -> Indices of the functional basis (integer vector)
%   ind.ba : Indices of the AR functional basis
%   ind.bc : Indices of the MA functional basis
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
%                                - '2sls' two stage least squares
%                                - 'ml' maximum likelihood
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
%           A.tsls, A.ms : Estimated projection parameters
%           obtained by the respective estimator class
%           A.time : AR parameters (projected in time)
%  C    -> Structure with the parameters of the model
%           C.tsls, C.ml : Estimated projection parameters
%           obtained by the respective estimator class
%           C.time : MA parameters (projected in time)
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
if nargin < 5
    fprintf('Incomplete information to estimate the model\n')
    return
else
    
    % Default basis function
    if ~isfield(options,'basis'), options.basis = 'sinus'; end
    
    % Default projection parameter estimator
    if ~isfield(options,'ParamEstim'), options.ParamEstim.Type = 'ols'; end
    
    % Default variance estimator
    if ~isfield(options,'VarEstim'), options.VarEstim.Type = 'c'; end
    
    % Assigning default values for the 'ml' and '2sls' parameter estimation methods
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
    
    if strcmp(options.ParamEstim.Type,'2sls')
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
    
    % Going to the FS-TARMA estimator
    [A,C,Se,e,criteria] = estimate(y,na,nc,ind,options);
end

%--------------------------------------------------------------------------
function [A,C,Se,e,criteria] = estimate(y,na,nc,ind,options)

% Extracting information from the input
pa = numel(ind.ba);
pc = numel(ind.bc);
N = length(y);

%-- Step 1 : Computing a long FS-TAR model --------------------------------
if ~isfield(options,'IniE')
    fprintf('Computing a long FS-TAR model...\n')
    nl = min(2*( na + nc ),40);                                                 % Model order of the long FS-TAR model
    opt_longTAR = options;                                                      % Copy the options provided for the estimation method
    opt_longTAR.ParamEstim.Type = 'ms';                                         % Use the multi stage method for the estimation of the long TAR model
    [~,~,e] = fstar(y,nl,ind,opt_longTAR);                                      % Estimate the long FS-TAR model and extract the estimation residuals
    e = [zeros(nl,1); e'];
    fprintf('Done!!!\n')
else
    e = options.IniE;
end

% Estimate the innovations variance
options.VarEstim.ind = ind.bs;
options.VarEstim.basis = options.basis;
Se = tdvar(e',options.VarEstim);

%-- Step 2 : Compute the FS-TARMA model -----------------------------------
fprintf('Computing the FS-TARMA model...\n')
% Computing the basis and the regression matrix
Gba = basis(N,ind.ba,options.basis);                                        % AR Basis
Gbc = basis(N,ind.bc,options.basis);                                        % MA Basis
Phi = RegMatrix(y,e,Gba,Gbc,na,nc,ind.ba,ind.bc,N);                         % Regression matrix

% Compute the WLS projection parameter estimate
w = Se.time(na+1:end);                                                      % Weights for the WLS method
[theta,e,criteria] = wls(Phi,y(na+1:end)',w);                               % WLS estimate of the parameters
A.tsls = theta(1:na*pa);                                                    % AR projection parameters
C.tsls = theta(na*pa+1:end);                                                % MA projection parameters
A.time = reshape(A.tsls,pa,na)'*Gba;                                        % AR parameters in time
C.time = reshape(C.tsls,pc,nc)'*Gbc;                                        % MA parameters in time

% Update the time-dependent innovations variance
Se = tdvar(e,options.VarEstim);

% Computing the criteria for the modeling performance
N = length(e);
criteria.rss = sum(e.^2);
criteria.rss_sss = criteria.rss/sum(y.^2);
criteria.mse = mean(e.^2);
criteria.logL = loglikelihood(e,Se.time);
criteria.bic = - criteria.logL + log(N)*(na*pa+nc*pc)/2;
criteria.spp = (na*pa+nc*pc)/N;
criteria.CondPhi = cond(Phi*Phi');

%--------------------------------------------------------------------------
function Phi = RegMatrix(y,e,Gba,Gbc,na,nc,ind_ba,ind_bc,N)
% Computes the regression matrix of the FS-TAR model based on the output
% and the basis

pa = numel(ind_ba);
pc = numel(ind_bc);
Phi = zeros(na*pa+nc*pc,N-na);

for i=1:na
    Phi((i-1)*pa+1:i*pa,:) = Gba(:,na+1:N).*repmat(y(na+1-i:N-i),pa,1);
end

for i=1:nc
    Phi(na*pa+(i-1)*pc+1:na*pa+i*pc,:) = Gbc(:,na+1:N).*repmat(e(na+1-i:N-i)',pc,1);
end