function [y,w,criteria] = SimulateTARMA(a,c,sw)
%--------------------------------------------------------------------------
% Function to simulate a realization of a TARMA model from given parameter
% trajectories "a" and "c", and innovations variance "sw"
% Input:
%       a   : AR parameter trajectories (na x N)
%       c   : MA parameter trajectories (nc x N)
%       sw  : Time-dependent innovations variance (1 x N)
% Output:
%       y   : Realization of the TARMA process (1 x N)
%       w   : Innovations sequence (1 x N)
%       criteria : Structure containing various performance criteria
%           criteria.rss        -> Residual Sum of Squares
%           criteria.rss_sss    -> Residual Sum of Squares over Series Sum of Squares
%           criteria.logL       -> Log likelihood
%
% Created by : David Avendano - January 2015
%--------------------------------------------------------------------------

[na,N] = size(a);
nc = size(c,1);

% Creating a realization of the innovations
w = sqrt(sw).*randn(1,N);

% Creating a realization of the process
y = zeros(size(w));
if nc == 0 % TAR model
    for tt=na+1:N
        y(tt) = -dot(a(:,tt),y(tt-1:-1:tt-na)) + w(tt);
    end
else        % TARMA model
    for tt=na+1:N
        y(tt) = -dot(a(:,tt),y(tt-1:-1:tt-na)) + dot(c(:,tt),w(tt-1:-1:tt-nc)) + w(tt);
    end
end

% Computing the performance criteria
criteria.rss = sum(w);
criteria.rss_sss = criteria.rss/sum(y);
criteria.logL = -N*log(2*pi)/2 - sum( log(sw) + w.^2 ./ sw );