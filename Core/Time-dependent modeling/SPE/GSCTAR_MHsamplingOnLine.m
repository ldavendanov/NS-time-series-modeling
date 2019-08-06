function [theta_hat,P_theta,y_hat,HyperPar,sample,other] = GSCTAR_MHsamplingOnLine(y,na,q,Nsample,options)
%--------------------------------------------------------------------------
% This function computes MAP estimates of the hyperparameters of a
% GSC-TAR model, including the stochastic constraint parameters, the
% innovations variance and the parameter innovations covariance matrix via
% Metropolis-Hastings random samping. It also provides the MAP parameter
% trajectories using the KF based on the optimized hyperparameters.
% Input parameters:
%   y       -> Observation vector ( vector : 1 x N )
%   na      -> Model order ( scalar )
%   q       -> Stochastic constraint order ( scalar )
%   Nsample -> Number of samples to evaluate ( scalar )    
%   options -> Estimator options
%       options.ShowProgress    : Flag to show progress of the algorithm (0-1)
%       options.M               : Size of the window to update the algorithm
%
% Output parameters:
%   theta_hat -> Estimated parameter trajectories ( matrix : na x N )
%   P_theta   -> (Smoothed) parameter estimation error covariance ( 3D matrix : na x na x N )
%   y_hat     -> One-step-ahead signal predictions ( vector : 1 x N )
%   HyperPar  -> Structure with the hyperparameters
%
%--------------------------------------------------------------------------

if nargin == 4
    options.ShowProgress = 1;
    options.M = 250;
end
M = options.M;
N = length(y);
Nsig = floor(N/M);
Nepoch = floor(Nsample/Nsig);

% Initializing the computing matrices
p_mu = zeros(q-1,Nsample);
sigma_v2 = zeros(1,Nsample);
sigma_w2 = zeros(1,Nsample);
logL = zeros(1,Nsample);

% Providing initial values
p_mu(:,1) = ones(q-1,1);
sigma_v2(1) = 1e-4;
sigma_w2(1) = 1e-3;
a = arburg(y(1:100),na);
theta0 = a(2:end);

% Providing the values of the proposal distributions
beta_v = 2*sigma_v2(1);
alpha_v = sigma_v2(1)/beta_v;
beta_w = 2*sigma_w2(1);
alpha_w = sigma_w2(1)/beta_w;
sigma_mu2 = 0.025;

% Initial value of the likelihood
mu = poly([1; p_mu(:,1)]);
HyperPar.mu = mu(2:end);
HyperPar.theta0 = theta0';
HyperPar.Sigma_v = sigma_v2(1)*eye(na);
HyperPar.sigma_w2 = sigma_w2(1);
options.estim = 'kf';
[~,~,~,other] = GSCTAR_MAPtrajectory(y(1:M),na,HyperPar,options);
logL(1) = -other.logL;

% Performing the Metropolis-Hastings sampling
clr = lines(4);
fprintf('Performing MCMC sampling\n')
for j=1:Nepoch,
    if j==1, ini = 2; else ini=1; end
    for k=ini:Nsig,
        
        fprintf('Sample No. %4d\n',k+Nsig*(j-1))
        
        % Obtaining a new sample from the proposal distribution
        l = k+Nsig*(j-1);
        sigma_v2(l) = gamrnd(alpha_v,beta_v);
        sigma_w2(l) = gamrnd(alpha_w,beta_w);
        
        cond = true;
        while cond
            p_mu(:,l) = p_mu(:,l-1) + sigma_mu2*randn(q-1,1);
            cond = max(abs(p_mu(:,l)) > 1);
        end
        
        % Evaluating the probability of the random variables
        q_v = gampdf( sigma_v2(l-(0:1)), alpha_v, beta_v );
        q_w = gampdf( sigma_w2(l-(0:1)), alpha_w, beta_w );
        q_mu = mvnpdf( p_mu(:,l), p_mu(:,l-1), sigma_mu2*eye(q-1) );
        q_mu(2) = mvnpdf( p_mu(:,l-1), p_mu(:,l), sigma_mu2*eye(q-1) );
        
        % Evaluating the likelihood for the old point
        mu = poly([1; p_mu(:,l-1)]);
        HyperPar.mu = mu(2:end);
        HyperPar.theta0 = theta0';
        HyperPar.Sigma_v = sigma_v2(l-1)*eye(na);
        HyperPar.sigma_w2 = sigma_w2(l-1);
        options.estim = 'kf';
        ind = M*(k-1)+(1:M);
        [~,~,~,other] = GSCTAR_MAPtrajectory(y(ind),na,HyperPar,options);
        logL_old = -other.logL;
        
        % Evaluating the likelihood for the new point
        mu = poly([1; p_mu(:,l)]);
        HyperPar.mu = mu(2:end);
        HyperPar.theta0 = theta0';
        HyperPar.Sigma_v = sigma_v2(l)*eye(na);
        HyperPar.sigma_w2 = sigma_w2(l);
        [~,~,~,other] = GSCTAR_MAPtrajectory(y(ind),na,HyperPar,options);
        logL_new = -other.logL;
        
        % Evaluating the ratio
        num = logL_new + log(q_v(2)) + log(q_w(2)) + log(q_mu(2));
        den = logL_old + log(q_v(1)) + log(q_w(1)) + log(q_mu(1));
        r = min(num-den,0);
        
        % Accept/reject
        if r<log(1e-6)
            logL(l) = logL_old;
            sigma_v2(l) = sigma_v2(l-1);
            sigma_w2(l) = sigma_w2(l-1);
            p_mu(:,l) = p_mu(:,l-1);
        else
            logL(l) = logL_new;
        end
        
        if options.ShowProgress == 1
            subplot(221)
            for m=1:q-1
                plot(l,p_mu(m,l),'.','Color',clr(m,:))
                hold on
            end
            ylabel('$\mu$','interpreter','latex')
            
            subplot(222)
            semilogy(l,sigma_w2(:,l),'.b')
            hold on
            ylabel('$\sigma_w^2$','interpreter','latex')
            
            subplot(223)
            semilogy(l,sigma_v2(:,l),'.b')
            ylabel('$\sigma_v^2$','interpreter','latex')
            hold on
            
            subplot(224)
            plot(l,logL(l),'.b')
            ylabel('$\ln \mathcal{L}$','interpreter','latex')
            hold on
            
            drawnow
        end
        
    end
end
fprintf('Done!!\n')

% Evaluating MAP estimates of the hyperparameters from the sample
mu_hat = poly([1; p_mu(:,l)]);
mu_hat = mu_hat(2:end);
sigma_v2_hat = sigma_v2(l);
sigma_w2_hat = sigma_w2(l);
HyperPar = [mu_hat(2:end) theta0 log(sigma_w2_hat) log(sigma_v2_hat)]';

% Computing for the optimized parameter values
options.estim = 'ks';
mu = poly([1; p_mu(:,l)]);
HP.mu = mu(2:end);
HP.theta0 = theta0';
HP.Sigma_v = sigma_v2(l)*eye(na);
HP.sigma_w2 = sigma_w2(l);
[theta_hat,P_theta,y_hat,other] = GSCTAR_MAPtrajectory(y,na,HP,options);

% Extracting the output values
sample.p_mu = p_mu;
sample.sigma_v2 = sigma_v2;
sample.sigma_w2 = sigma_w2;
sample.logL = logL;