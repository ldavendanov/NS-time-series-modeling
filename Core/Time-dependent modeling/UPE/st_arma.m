function [A,C,E,TT,CRITERIA] = st_arma(Y,NN,n_win,t_step)
%--------------------------------------------------------------------------
%st_arma Computes estimates of a TARMA model via short time windows.
%   [A,C,E,Tst,CRITERIA] = st_arma(Y,na,nc,n_win,t_step)
%
%   Y       : The time series data (1 x N)
%   NN      : NN = [na nc], The AR and MA orders 
%   n_win   : Window length
%   t_step  : Number of samples between succesive windows
%
%   A       : AR time-dependent matrix (na x N)
%   C       : MA time-dependent matrix (na x N)
%   E       : Residual sequence vector (1 x N)
%   TT      : Time vector corresponding to the samples of the parameters (1 x N)
%   CRITERIA : Structure with fields
%              RSS : Residual Sum of Squares
%              RSS_SSS : Normalized Residual Sum of Squares
%              BIC : Bayesian Information Criterion
%--------------------------------------------------------------------------
%   M. Spiridonakos 2005
%   Copyright 1980-2005 The MinaS, Inc.
%   $ Version: 1.1 $  $ Date: 2005 $
%--------------------------------------------------------------------------

N = length(Y);
na = NN(1); nc = NN(2);
est_param=na+nc;

TT= ((n_win-1)/2)+1:t_step:N-((n_win-1)/2);
rr=((n_win-1)/2);

max_n=max(na,nc);
A=zeros(length(TT),max_n+1);
C=zeros(length(TT),max_n+1);
E=zeros(n_win,length(TT));
criteria.rss=zeros(length(TT),1);
criteria.var_w=zeros(length(TT),1);
criteria.rss_sss=zeros(length(TT),1);
criteria.BIC=zeros(length(TT),1);

qq=0;
if nc == 0
    for ii=TT
        qq=qq+1;
        DAT = iddata(Y(ii-rr:ii+rr)',[]);
        MM = ar(DAT,na);
        [C(qq,:),A(qq,:)] = tfdata(MM,'v');
        preds = predict(MM,Y(ii-rr:ii+rr)',1);
        E(:,qq) = Y(ii-rr:ii+rr)-preds';
        criteria.rss(qq) = norm(E(:,qq))^2;
        criteria.var_w(qq) = var(E(30:end,qq));
        criteria.rss_sss(qq) = norm(E(:,qq))^2/norm(Y(ii-rr:ii+rr))^2;
        criteria.BIC(qq) = log(criteria.rss(qq)/n_win)+est_param*(log(n_win)/n_win);
    end
else
    for ii=TT
        qq=qq+1;
        DAT = iddata(Y(ii-rr:ii+rr),[]);
        MM = armax(DAT,[na nc]);
        [C(qq,:),A(qq,:)] = tfdata(MM,'v');
        preds = predict(MM,Y(ii-rr:ii+rr),1);
        E(:,qq) = Y(ii-rr:ii+rr)-preds;
        criteria.rss(qq) = norm(E(:,qq))^2;
        criteria.var_w(qq) = var(E(30:end,qq));
        criteria.rss_sss(qq) = norm(E(:,qq))^2/norm(Y(ii-rr:ii+rr))^2;
        criteria.BIC(qq) = log(criteria.rss(qq)/n_win)+est_param*(log(n_win)/n_win);
    end
end
A = A(:,2:end)';
C = C(:,2:end)';

CRITERIA.rss=mean(criteria.rss);
CRITERIA.rss_sss=mean(criteria.rss_sss);
CRITERIA.BIC=mean(criteria.BIC);
