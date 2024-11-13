clear
close all
clc

syms t
syms M m Jo
syms kh kv kt
syms l
syms x(t) y(t) theta(t) psi(t)

ox = x + l*cos( theta + psi );
oy = y + l*sin( theta + psi );

x_dot = diff(x,t);
y_dot = diff(y,t);
theta_dot = diff(theta,t);
psi_dot = diff(psi,t);
ox_dot = diff(ox,t);
oy_dot = diff(oy,t);

T = M*x_dot.^2/2 + M*y_dot.^2/2 + m*ox_dot.^2/2 + m*oy_dot.^2/2 + Jo*psi_dot.^2/2;
V = (kh*x.^2)/2 + (kv*y.^2)/2 + (kt*theta.^2)/2;

L = T-V;

A1 = diff( diff(L,x_dot), t );
A2 = diff( L, x );

eom1 = simplify( expand( A1 - A2 ) );

B1 = diff( diff(L,y_dot), t );
B2 = diff( L, y );

eom2 = simplify( expand( B1 - B2 ));

C1 = diff( diff(L,theta_dot), t );
C2 = diff( L, theta );

eom3 = simplify( C1 - C2 );

D1 = diff( diff(L,psi_dot), t );
D2 = diff( L, psi );

eom4 = simplify( D1 - D2 );