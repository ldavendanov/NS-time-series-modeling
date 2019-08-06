function [a,c,se,other] = sptarmaVar(obs,order,options,variances)

options.P.sigma_w = variances(1);
options.P.Sigma_v = variances(2);
[a,c,se,other] = sptarma(obs,order,options);