function dx = DuffingEq(t,x,F,theta)

if nargin < 4
    alpha = 2;
    beta = 4e-1;
    delta = 0.1;
    gamma = 10;
else
    alpha = theta(1);
    beta = theta(2);
    gamma = theta(3);
    delta = theta(4);
end

dx(1) = x(2);
dx(2) = gamma*F(t) - delta*x(2) - beta*x(1)^3 - alpha*x(1);
dx = dx(:);