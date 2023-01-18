function [A,C,S,res,criteria,P,Phi,Psi,thm] = rmlvtarma(z,nn,ff,options)
%--------------------------------------------------------------------------
% RMLTVARMA Computes estimates recursively for an TVARMA model
% via the Recursive Maximum Likelihood method.
%   [A,C,S,E,CRITERIA] = RMLTVARMA(Z,NN,FF,OPTIONS)
%
%   Z: The output data (k x N)
%   NN : NN=[na nc ni], The AR and MA orders, and inverse function order
%       (used for the MA polynomial matrix stabilization)
%   FF :  Forgetting factor (0< ff <1; scalar)
%   OPTIONS : (structure)
%              TH0 : Initial value of parameters
%              P0 : Initial "covariance" matrix
%              PHI0 : Initial PHI regression matrix
%              PSI0 : Initial PSI auxiliary matrix
%              Res_t0 : number of initial resiuals excluded from the criteria computation
%              M : window length = 2*M+1
%
%   A : AR time-dependent matrix (k*N x k*na)
%   C : MA time-dependent matrix (k*N x k*nc)
%   S : Time-dependent Covariance matrix (k x k*N)
%   E: Residual sequence vector (k x N)
%   CRITERIA :
%              RSS : Residual Sum of Squares
%              RSS_SSS : Normalized Residual Sum of Squares
%   P : Last "covariance" matrix (k^2*(na+nc) x k^2*(na+nc))
%   PHI : Last PHI regression matrix (k^2*(na+nc) x k)
%   PSI : Last PSI auxiliary matrix (k^2*(na+nc) x k)
%   THM : Time-dependent parameter vector (k^2*(na+nc) x 1)
%--------------------------------------------------------------------------
%   Copyright 1980-2007 The MinaS, Inc.
%   $ Version: 1.1 $  $ Date: 2007/02/28 $
%--------------------------------------------------------------------------

[k,N]=size(z);

na = nn(1); nc = nn(2); ni= nn(3);
n = max([na nc]);

% Parameter vector length
d = (k^2)*(na + nc);

Aindex = 1:(k^2)*na;
Cindex = (k^2)*na+1:(k^2)*(na+nc);
A = zeros(k*N,k*na);
C = zeros(k*N,k*nc);

if nargin<4, Psi=zeros(d,k); Phi=zeros(d,k);  P=10000*eye(d);
    th=eps*ones(d,1); thm=eps*ones(d,1); M = 256; res_t0 = 100;
else
    if ~(isfield(options,'Psi')),        options.Psi = zeros(d,k);      end
    if ~(isfield(options,'Phi')),        options.Phi = zeros(d,k);      end
    if ~(isfield(options,'P0')),         options.P0 = 10000*eye(d);     end
    if ~(isfield(options,'th0')),        options.th0 = eps*ones(d,1);   end
    if ~(isfield(options,'M')),          options.M = 256;               end
    if ~(isfield(options,'res_t0')),     options.res_t0 = 100;          end
    if length(options.th0)~=d, error('The length of th0 must equal the number of estimated parameters!'),end
    Phi = options.Phi;
    Psi = options.Psi;
    P = options.P0;
    th = options.th0;
    M = options.M;
    res_t0 = options.res_t0;
end
lam = ff;

yhat = zeros(k,N);
[zf,ef]=deal(zeros(k,nc));
for t = 1:N
    % Prediction error
    yh = Phi'*th;   epsi = z(:,t)-yh;
    
    % Gain
    K = P*Psi*(inv(lam*eye(k) + Psi'*P*Psi));
    
    % Covariance update
    P = (P-K*Psi'*P)./lam;
    
    % Estimator update
    th = th+K*epsi;
    
    % Stabilization procedure
    CB = [eye(k) reshape(th(Cindex),k,k*nc)];
    cij = cell(k,k);
    for ii=1:k
        for jj=1:k
            cij{ii,jj}=CB(ii,jj:k:end);
        end
    end
    detC = celldet(cij);
    abs_r = abs(roots(detC));
    if ~isempty(find(abs_r >= 1,1))
        % Stabilization via Yule Walker modified equations
        [CBstab]=stabYW(CB(:,k+1:end),nc,k,ni);
        % Verification of the stabilization
        CBver = [eye(k) CBstab];
        cij = cell(k,k);
        for ii=1:k
            for jj=1:k
                cij{ii,jj}=CBver(ii,jj:k:end);
            end
        end
        detCv = celldet(cij);
        abs_r = abs(roots(detCv));
        if ~isempty(find(abs_r >= 1,1))
            warning('stabilization failed (t= %d)!!!',t);
        end
    else
        CBstab = CB(:,k+1:end);
    end
    
    
    % Stabilized theta
    th(Cindex)=reshape(CBstab,(k^2)*nc,1);
    
    % A-posteriori error
    epsilon = z(:,t)-Phi'*th;
    
    % Regressor matrix update
    if nc == 0,
        Phi = [kron(-z(:,t),eye(k)); Phi(1:k^2*(na-1),:)];
    else
        Phi = [kron(-z(:,t),eye(k)); Phi(1:k^2*(na-1),:);...
            kron(epsilon,eye(k)); Phi((k^2)*na+1:k^2*(na+nc-1),:)];
    end
    
    % Filtering
    zf = [(z(:,t)-CBstab*reshape(zf(:,1:nc),k*nc,1)), zf(:,1:nc-1)];
    ef = [(epsilon-CBstab*reshape(ef(:,1:nc),k*nc,1)), ef(:,1:nc-1)];
    
    % Psi matrix update
    if nc == 0
        Psi = [kron(-zf(:,1),eye(k)); Psi(1:k^2*(na-1),:)];
    else
        Psi = [kron(-zf(:,1),eye(k)); Psi(1:k^2*(na-1),:);...
            kron(ef(:,1),eye(k)); Psi((k^2)*na+1:k^2*(na+nc-1),:)];
    end
    % AR and MA time-dependent matrices
    A((t-1)*k+1:t*k,:) = reshape(th(Aindex),k,k*na);
    C((t-1)*k+1:t*k,:) = reshape(th(Cindex),k,k*nc);
    
    % Prediction
    yhat(:,t)= yh;
    
    thm(:,t) = th;
end


% Residuals computation
res = z-yhat;

% Covariance Matrix Estimation
S = zeros(k,k*N);
for ii=n+M+1:N-M
    S(:,(ii-1)*k+1:ii*k) = (res(:,ii-M:ii+M)*(res(:,ii-M:ii+M).'))/(2*M+1);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                          Criteria Computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   SSS
sss = sum(sum(z(:,res_t0:end).^2));

%   RSS - RSS/SSS
criteria.rss = sum(sum(res(:,res_t0:end).^2));
criteria.rss_sss = 100*(criteria.rss/sss);
A = A';
C = C';



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    Stability (Yule Walker)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A]=stabYW(A,na,s,ni)
%_______________________________________________________________________________________________
%	This function stabilizes the A(B) polynomial, using modified Yule-Walker equations (Ref.1)
%
%	A(B):The matrix polynomial  A(B) = I + A1(B) + ... + Ana(B)  (s x (na+1)s)
%
%	s   :The number of the outputs
%
%	na  :The order of the A(B) polynomial
%
%	ni  :The order of the least squares inverse (Ref.1)
%_______________________________________________________________________________________________
%	ATTENTION: The unstable A(B) polynomial should be given in the following form:
%
%			A(B)=[A1 A2 ... Ana]
%
%	The matrix coefficients, A1 ... Ana, should be square matrices with dimensions s x s
%
%	The stabilized A(B) is calculated also in the previous form  (A(B)=[A1 A2 ... Ana])
%	but the real A(B) is: A(B)= I + A1(B) + ... + Ana(B)
%_______________________________________________________________________________________________
%
%	Ref.1	: ''MIMO LMS-ARMAX IDENTIFICATION OF VIBRATING STRUCTURES:
%				- A Critical Assessment -
%				A. Florakis & S.D.Fassois
%_______________________________________________________________________________________________

A1=[];
for i=1:na
    A1=[A1; (A(:,(i-1)*s+1:i*s))'];
end
A=[eye(s) A];
A1=[eye(s); A1];
A2=A;

% First stage:	Q(B)*A(B)=I (Ref.1)
R=[];

% Creation a part of the autocorrelation matrix R=[R(0); R(1); ... ;R(ni-1)] (Ref.1)
for k=1:ni
    if k<=na+1;
        RR=A2*A1;
        A2(:,size(A2,2)-s+1:size(A2,2))=[];
        A1(1:s,:)=[];
    else
        RR=zeros(s);
    end
    R=[R; RR];
end
RRRR=R;

% Creation the of matrix Rx=[ R(0)     R(-1)   ...  R(-ni+1)
%						      R(1)     R(0)    ...  R(-ni+2)
%                              .         .           .
%						       .         .           .
%							   .         .           .
%						     R(ni-1)  R(ni-2)  ...  R(0)  ]  (Ref.1)
for k=2:ni
    kk=zeros((k-1)*s,s);
    RRR=[kk; R];
    RRR(size(RRR,1)-(k-1)*s+1:size(RRR,1),:)=[];
    RRRR=[RRRR RRR];
end
d=(tril(RRRR,-1))';
Rxx=tril(RRRR)+d;

% Creation of the matrix r=[R(1); R(2); ... ;R(ni)] (Ref.1)
rxx=Rxx(:,1:s);
rxx(1:s,:)=[];
rxx=[rxx; zeros(s)];

% Estimation of the Q(B) poly:	Rx * Q'(B) = -rx
Q=-Rxx\rxx;
Q=[eye(s); Q];
Q1=Q';
clear A1 A2 RRRR RRR R rxx k kk d

% Second stage: A(B)*Q(B)=I, Estimation of the stabilized A(B) poly
R=[];
for k=1:na
    RR=Q1*Q;
    Q1(:,size(Q1,2)-s+1:size(Q1,2))=[];
    Q(1:s,:)=[];
    R=[R; RR];
end
r2=Q1*Q;
RRRR=R;
for k=2:na
    kk=zeros((k-1)*s,s);
    RRR=[kk; R];
    RRR(size(RRR,1)-(k-1)*s+1:size(RRR,1),:)=[];
    RRRR=[RRRR RRR];
end
d=(tril(RRRR,-1))';
Rxx=tril(RRRR)+d;
rxx=Rxx(:,1:s);
rxx(1:s,:)=[];
rxx=[rxx; r2];
A1=-Rxx\rxx;
A=[];
for i=1:na
    A=[A (A1((i-1)*s+1:i*s,:))']; % The stabilized polynomial A(B) = [A1 A2 ... Ana]
end
clear Q


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                 Cell Determinant
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function detc = celldet(a)

% CELLDET computes the determinant of a polynomial matrix which
% must be given in a cell form (see function cell.m).
% Returns the determinant of a rectangular cell
% containing polynomials (coefficient vectors) as elements.
%
% INPUT
% A : the polynomial matrix.
%
% OUTPUT
% DETC : the determinant of matrix A.

[N,M] = size(a);
if N ~= M
    disp('ERROR: Not rectangular cell.')
    return
end

switch length(a)
    case 1
        detc = a{1,1};
    case 2
        detc = conv(a{1,1},a{2,2}) - conv(a{1,2},a{2,1});
    otherwise
        detc = 0;
        for i = 1:N
            [ac{1:N-1,1:N-1}] = deal(a{2:N,[1:i-1 i+1:N]});
            detc = detc + (-1)^(i+1)*(conv(a{1,i},celldet(ac)));
        end
end
