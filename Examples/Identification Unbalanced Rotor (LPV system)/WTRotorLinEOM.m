function dz = WTRotorLinEOM( t,z, System, OmegaRef, TauM, fs )
% Linearized EOM of a simplified wind turbine model

[M,C,K] = WTSystemMatrices( System, OmegaRef, t );

A = [zeros(5) eye(5);
     -M\[K C]];
b = [ zeros(5,1);
      M\WTLoadVector( System, OmegaRef, t, TauM, fs )];

dz = A*z + b;

end

