function Gb = basis(N,ind,options)
%--------------------------------------------------------------------------
% Function to compute a functional basis of lenght 'N' for the indices
% summarized in the vector 'ind'.
% The basis type and other parameters are within the 'options' structure
% Created by : David Avendano - April 2013 - Ver 1.0
%               All rights reserved
%--------------------------------------------------------------------------

% Initializing the basis matrix
pa = ind(end);
Gb = zeros(pa,N);
T = 0:N-1;

% Computing the basis
switch options.type
    
    %-- Fourier basis
    case 'sinus'
        if isfield(options,'basis_freq')
            fo = options.basis_freq;
        else
            fo = 1/N;
        end
        
        % The basis order must be odd when using the Fourier basis to
        % ensure numerical stability
        if mod(pa,2) == 0
            pa = pa-1;
            warning('Basis order must be odd when using the Fourier basis. Basis order set to p-1')
        end
        
        Gb = ones(pa,N);
        for j=1:(pa-1)/2
            Gb(2*j,:) = sin(j*2*pi*fo*T);
            Gb(2*j+1,:) = cos(j*2*pi*fo*T);
        end
        
    %-- Hermite polynomials
    case 'hermite'
        T = (0:N-1)/(N-1);
        Gb = ones(pa,N);
        Gb(2,:) = 2*T;
        for j=3:pa
            Gb(j,:) = 2*T.*Gb(j-1,:) - 2*(j-1)*Gb(j-2,:);
        end
        
    %-- DCT basis
    case 'cosine'
        if isfield(options,'basis_freq')
            fo = options.basis_freq;
        else
            fo = 1/N;
        end
        
        Gb = ones(pa,N);
        for k=1:pa
            Gb(k,:) = cos( k*2*pi*fo*T );
        end
        
    %-- Polynomial basis (Chebyshev polynomials)
    case 'poly'
        T = (0:N-1)/(N-1);
        Gb = ones(pa,N);
        Gb(2,:) = T;
        for k=3:pa
            Gb(k,:) = 2*T.*Gb(k-1,:) - Gb(k-2,:);
        end
        
    % B-spline basis
    case 'bspline'
        t = 0:N-1;
        k = pa;
        n = pa;
        
        m = n + k;                % number of knots
        step = (N-1)/(m-2*k+1);   % Internal knots distance
        tindex = [zeros(1,k-1) round(0:step:N-1) (N-1)*ones(1,k-1)]; % knots
        
        b = zeros(m,k,N);
        for j = 1:m-1
            b(j,1,t >= tindex(j) & t < tindex(j+1)) = 1 ;
        end
        
        omega = zeros(m-1,k,N);
        for i = 2:k
            for j = 1:m-i+1
                if tindex(j) < tindex(j+i-1)
                    omega(j,i,:) = (t-tindex(j))/(tindex(j+i-1)-tindex(j));
                end
            end
        end
        
        for i = 2:k
            for j = 1:m-i
                b(j,i,:) = omega(j,i,:).*b(j,i-1,:) + (1 - omega(j+1,i,:)).*b(j+1,i-1,:);
            end
        end
        
        Gb = squeeze(b(1:n,k,:));
        Gb(end,end) = 1;
end

% Extracting the required basis indexes
Gb = Gb(ind,:);