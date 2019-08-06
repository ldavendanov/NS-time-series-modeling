function [theta_hat,P_theta,y_hat,HyperPar,sample,other] = GSCTARMA_MHsampling(y,na,nc,q,Nsample,options)
%--------------------------------------------------------------------------
% This function computes MAP estimates of the hyperparameters of a
% GSC-TAR model, including the stochastic constraint parameters, the
% innovations variance and the parameter innovations covariance matrix via
% Metropolis-Hastings random samping. It also provides the MAP parameter
% trajectories using the KF based on the optimized hyperparameters.
% Input parameters:
%   y       -> Observation vector ( vector : 1 x N )
%   na      -> Model order ( scalar )
%   nc      -> Model order ( scalar )
%   q       -> Stochastic constraint order ( scalar )
%   Nsample -> Number of samples to evaluate ( scalar )    
%   options -> Estimator options
%
% Output parameters:
%   theta_hat -> Estimated parameter trajectories ( matrix : na+nc x N )
%   P_theta   -> (Smoothed) parameter estimation error covariance ( 3D matrix : na+nc x na+nc x N )
%   y_hat     -> One-step-ahead signal predictions ( vector : 1 x N )
%   HyperPar  -> Structure with the hyperparameters
%
%--------------------------------------------------------------------------

% Initializing the computing matrices
p_mu = zeros(q,Nsample);
sigma_v2 = zeros(1,Nsample);
sigma_w2 = zeros(1,Nsample);
logL = zeros(1,Nsample);

% Providing initial values
p_mu(:,1) = ones(q,1);
sigma_v2(1) = 1e-6;
sigma_w2(1) = 1e-3;

% Initial parameter values
data = iddata(y',[],1);
sys = armax(data,[na nc]);
theta0 = [sys.a(2:end) sys.c(2:end)];

% Providing the values of the proposal distributions
beta_v = sigma_v2(1)/2;
alpha_v = sigma_v2(1)/beta_v;
beta_w = sigma_w2(1)/2;
alpha_w = sigma_w2(1)/beta_w;
sigma_mu2 = 0.1;

% Initial value of the likelihood
mu = poly(p_mu(:,1));
HyperPar.mu = mu(2:end);
HyperPar.theta0 = theta0';
HyperPar.Sigma_v = sigma_v2(1)*eye(na+nc);
HyperPar.sigma_w2 = sigma_w2(1);
options.estim = 'kf';
[~,~,~,other] = GSCTARMA_MAPtrajectory(y,na,nc,HyperPar,options);
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
        p_mu(:,k) = p_mu(:,k-1) + sigma_mu2*randn(q,1);
        cond = max(abs(p_mu(:,k)) > 1);
    end
    
    % Evaluating the probability of the random variables
    q_v = gampdf(sigma_v2(k-(0:1)),alpha_v,beta_v);
    q_w = gampdf(sigma_w2(k-(0:1)),alpha_w,beta_w);
    q_mu = mvnpdf(p_mu(:,k),p_mu(:,k-1),sigma_mu2*eye(q));
    q_mu(2) = mvnpdf(p_mu(:,k-1),p_mu(:,k),sigma_mu2*eye(q));
    
    % Evaluating the likelihood
    mu = poly(p_mu(:,k));
    HyperPar.mu = mu(2:end);
    HyperPar.theta0 = theta0';
    HyperPar.Sigma_v = sigma_v2(k)*eye(na+nc);
    HyperPar.sigma_w2 = sigma_w2(k);
    options.estim = 'kf';
    [~,~,~,other] = GSCTARMA_MAPtrajectory(y,na,nc,HyperPar,options);
    logL(k) = -other.logL;
    
    % Evaluating the ratio
    num = logL(k)+log(q_v(2))+log(q_w(2))+log(q_mu(2));
    den = logL(k-1)+log(q_v(1))+log(q_w(1))+log(q_mu(1));
    r = min(num-den,0);
    
    % Accept/reject
    if r<log(0.4)
        logL(k) = logL(k-1);
        sigma_v2(k) = sigma_v2(k-1);
        sigma_w2(k) = sigma_w2(k-1);
        p_mu(:,k) = p_mu(:,k-1);
    end
    
    if options.ShowProgress == 1
        subplot(221)
        for l=1:q
            plot(k,p_mu(l,k),'.','Color',clr(l,:))
            hold on
        end
        
        subplot(222)
        semilogy(k,sigma_w2(:,k),'.b')
        hold on
        
        subplot(223)
        semilogy(k,sigma_v2(:,k),'.b')
        hold on
        
        subplot(224)
        plot(k,logL(k),'.b')
        hold on
        
        drawnow
    end
    
end
fprintf('Done!!\n')

% Evaluating MAP estimates of the hyperparameters from the sample
mu_hat = poly(median(p_mu,2));
mu_hat = mu_hat(2:end);
sigma_v2_hat = median(sigma_v2);
sigma_w2_hat = median(sigma_w2);
HyperPar = [mu_hat theta0 log(sigma_w2_hat) log(sigma_v2_hat)]';

% Computing for the optimized parameter values
options.estim = 'ks';
HP.mu = HyperPar(1:q)';                             % Optimized stochastic constraint vector
HP.theta0 = HyperPar(q+(1:na+nc));                    % Optimized parameter mean
HP.sigma_w2 = exp(HyperPar(end-1));                 % Optimized innovations variance
HP.Sigma_v = exp(HyperPar(end))*eye(na+nc);         % Optimized parameter innovations variance
[theta_hat,P_theta,y_hat,other] = GSCTARMA_MAPtrajectory(y,na,nc,HP,options);

% Extracting the output values
sample.p_mu = p_mu;
sample.sigma_v2 = sigma_v2;
sample.sigma_w2 = sigma_w2;
sample.logL = logL;