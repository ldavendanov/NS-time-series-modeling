function [theta_hat,P_theta,y_hat,HyperPar,sample,other] = GSCTAR_MHsampling(y,na,q,Nsample,options)
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
end

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
[~,~,~,other] = GSCTAR_MAPtrajectory(y,na,HyperPar,options);
logL(1) = -other.logL;

% Performing the Metropolis-Hastings sampling
clr = lines(4);
fprintf('Performing MCMC sampling\n')
for k=2:Nsample,
    
    fprintf('Sample No. %4d\n',k)
    
    % Obtaining a new sample from the proposal distribution
    sigma_v2(k) = gamrnd(alpha_v,beta_v);
    sigma_w2(k) = gamrnd(alpha_w,beta_w);
    
    cond = true;
    while cond
        p_mu(:,k) = p_mu(:,k-1) + sigma_mu2*randn(q-1,1);
        cond = max(abs(p_mu(:,k)) > 1);
    end
    
    % Evaluating the probability of the random variables
    q_v = gampdf(sigma_v2(k-(0:1)),alpha_v,beta_v);
    q_w = gampdf(sigma_w2(k-(0:1)),alpha_w,beta_w);
    q_mu = mvnpdf(p_mu(:,k),p_mu(:,k-1),sigma_mu2*eye(q-1));
    q_mu(2) = mvnpdf(p_mu(:,k-1),p_mu(:,k),sigma_mu2*eye(q-1));
    
    % Evaluating the likelihood
    mu = poly([1; p_mu(:,k)]);
    HyperPar.mu = mu(2:end);
    HyperPar.theta0 = theta0';
    HyperPar.Sigma_v = sigma_v2(k)*eye(na);
    HyperPar.sigma_w2 = sigma_w2(k);
    options.estim = 'kf';
    [~,~,~,other] = GSCTAR_MAPtrajectory(y,na,HyperPar,options);
    logL(k) = -other.logL;
    
    % Evaluating the ratio
    num = logL(k)+log(q_v(2))+log(q_w(2))+log(q_mu(2));
    den = logL(k-1)+log(q_v(1))+log(q_w(1))+log(q_mu(1));
    r = min(num-den,0);
    
    % Accept/reject
    if r<log(1e-12)
        logL(k) = logL(k-1);
        sigma_v2(k) = sigma_v2(k-1);
        sigma_w2(k) = sigma_w2(k-1);
        p_mu(:,k) = p_mu(:,k-1);
    end
    
    if options.ShowProgress == 1
        subplot(221)
        for l=1:q-1
            plot(k,p_mu(l,k),'.','Color',clr(l,:))
            hold on
        end
        ylabel('$\mu$','interpreter','latex')
        
        subplot(222)
        semilogy(k,sigma_w2(:,k),'.b')
        hold on
        ylabel('$\sigma_w^2$','interpreter','latex')
        
        subplot(223)
        semilogy(k,sigma_v2(:,k),'.b')
        ylabel('$\sigma_v^2$','interpreter','latex')
        hold on
        
        subplot(224)
        plot(k,logL(k),'.b')
        ylabel('$\ln \mathcal{L}$','interpreter','latex')
        hold on
        
        drawnow
    end
    
end
fprintf('Done!!\n')

% Evaluating MAP estimates of the hyperparameters from the sample
mu_hat = poly([1; p_mu(:,k)]);
mu_hat = mu_hat(2:end);
sigma_v2_hat = sigma_v2(k);
sigma_w2_hat = sigma_w2(k);
HyperPar = [mu_hat(2:end) theta0 log(sigma_w2_hat) log(sigma_v2_hat)]';

% Computing for the optimized parameter values
options.estim = 'ks';
mu = poly([1; p_mu(:,k)]);
HP.mu = mu(2:end);
HP.theta0 = theta0';
HP.Sigma_v = sigma_v2(k)*eye(na);
HP.sigma_w2 = sigma_w2(k);
[theta_hat,P_theta,y_hat,other] = GSCTAR_MAPtrajectory(y,na,HP,options);

% Extracting the output values
sample.p_mu = p_mu;
sample.sigma_v2 = sigma_v2;
sample.sigma_w2 = sigma_w2;
sample.logL = logL;