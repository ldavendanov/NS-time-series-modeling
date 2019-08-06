function y = SimulateTARX(a,c,x)
%--------------------------------------------------------------------------
% Function to simulate a realization of a TARX model from given parameter
% trajectories "a" and "c", and input "x"
% Input:
%       a   : AR parameter trajectories (na x N)
%       c   : MA parameter trajectories (nc x N)
%       x   : Input signal (1 x N)
% Output:
%       y   : Realization of the TARMA process (1 x N)
%
% Created by : David Avendano - January 2015
%--------------------------------------------------------------------------

[na,N] = size(a);
nc = size(c,1);

% Creating a realization of the process
y = zeros(size(x));
if nc == 0 % TAR model
    for tt=na+1:N
        y(tt) = -dot(a(:,tt),y(tt-1:-1:tt-na)) + x(tt);
    end
else        % TARMA model
    n0 = max(na,nc);
    for tt=n0+1:N
        y(tt) = -dot(a(:,tt),y(tt-1:-1:tt-na)) + dot(c(:,tt),x(tt-1:-1:tt-nc)) + x(tt);
    end
end