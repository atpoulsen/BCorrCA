function [A,Z,Elambda,Epsi,Ealpha,Lbound,U,converged] = BCorrCA(X,opts)
%BCorrCA Bayesian Correlated Component Analysis
%   [A,Z] = BCorrCA(X,opts) computes the correlated components, Z, and
%   their corresponding forward models, A, for the multiview input, X, as
%   described in [1]. The model estimates a shared forward model, U, and
%   estimates the similarity between it and A. The variable names follows 
%   the notation used in [1].
%
%   [1] Kamronn, S., Poulsen, A. T., & Hansen, L. K. (2015). Multiview
%   Bayesian Correlated Component Analysis. Neural Computation.
%
%   Andreas Trier Poulsen (atpo@dtu.dk) and Simon Kamronn.
%   Technical University of Denmark, Cognitive systems. - Oct. 2015
%
% Inputs
% X    - 3D matrix of observations [Input dimension, Samples, Views].
% opts - [optional. Can also be set as "[]" for default settings].
%        Struct containing fields that define customization options for 
%        BCorrCA, such as hyperparameters. It is possible to only define
%        some of the fields.
%  .threshold - convergence criterion. Defined as the relative change in
%               lower bound. Default: 1e-6.
%      .nIter - number of maximum iterations. Default: 100.
%          .K - maximum number of hidden sources. Default: Input dimension.
%      .lSize - flexibility of similarity with the common forward model U.
%               Given as string; 'Shared': One lambda parameter for all
%               components and views. 'K': One lambda for each
%               view, shared across views. 'KM': One lambda for each
%               component in each view. Default: 'Shared'.
%          .S - prior knowlegde of covariance structure for the noise. 
%               For use in the the Wishart distribution of the noise.
%               Default: Identity matrix.
%         .v0 - number of degrees of freedom for the Wishart distribution.
%               Default: Input dimension + 1.
%    .verbose - print status. 0 = no, 1 = prints if the algorithm converged
%               within the maximum number of iterations. 2 = Also prints
%               time spent and relative change in lower bound. Default: 2.
%      .nIter - no. of iterations between printing change in lower bound 
%               and time spent. Only relevant when verbose is 2.
%               Default: 10.
% 
% Outputs
% A         - Estimated forward model.
% Z         - Estimimated components.
% Elambda   - Estimated similarity hyperparameter.
% Epsi      - Estimated noise presicion matrix.
% Ealpha    - Estimated forward model hyperparameter.
% Lbound    - Lower bound for each iteration.
% U         - Shared forward model.
% converged - Indicates if the algorithm converged within the maximum
%             number of iterations. 

[D,N,M] = size(X);

%% Setting default setting
threshold 	= 	1e-6;
nIter 		= 	100;
K 			= 	D;
lSize		=	'Shared';
S0inv		=	1e-3*repmat(eye(D),[1,1,M]);
v0			=	D+1;
converged   =   1;
verbose     =   2;
printIter   =   10;

% Checking inputs
switch nargin
    case 2
        if isfield(opts,'threshold'),	threshold = opts.threshold; end
        if isfield(opts,'nIter'), 		nIter = opts.nIter; end
        if isfield(opts,'K'),           K = opts.K; end
        if isfield(opts,'lSize'), 		lSize = opts.lSize; end
        if isfield(opts,'v'), 			v0 = opts.v; end
        if isfield(opts,'S')
            if ndims(opts.S)==3
                S0inv = opts.S*v0;
            else
                S0inv = repmat(opts.S,[1,1,M])*v0;
            end
        end
        if isfield(opts,'verbose'), 	verbose = opts.verbose; end
        if isfield(opts,'printIter'), 	printIter = opts.printIter; end
    case 1
    otherwise
        error('Wrong number of input')
end


%% Initialising and preallocating parameters
% Hyperparameters
a0=1e-3;
b0=a0;
a_alpha = a0 + D/2;
if strcmp(lSize,'K')
    a_lambda = a0 + M*D/2;
elseif strcmp(lSize,'KM')
    a_lambda = a0 + D/2;
else
    a_lambda = a0 + M*D*K/2;
end
v_psi = N + v0;


% Variables
datavar=sum(sum(var(X,0,2)));
alpha = (M*D)./datavar;
A = zeros(D,K,M);
for m = 1:M;
    A(:,:,m) = randn(D,K)*chol(diag(1./alpha)); % omregner varians og derefter std.
end
Z = zeros(K,N);
Zspec = zeros(K,N,M);
lambda_ini = 1; 
U = randn(D,K)*chol(diag(1./lambda_ini));


% Moments
Epsi = zeros(D,D,M);
for m = 1:M;
    Epsi(:,:,m)=a0*eye(D)/S0inv(:,:,m);
end
Eaa = zeros(K,M);
XX = zeros(D,D,M);
for m = 1:M;
    XX(:,:,m)=X(:,:,m)*X(:,:,m)';
end
Sig_ad=zeros(K,K,D,M);
for m=1:M
    for d=1:D
        Sig_ad(:,:,d,m) = b0/a0 *eye(K);
    end
end

Ealpha = alpha*ones(K,1);
Elambda = lambda_ini*ones(K,M);

%% Updating variables in iterations
tstart=tic;
Iter =1;
run=1;

while run   
    if verbose>=2 && Iter==round(Iter/printIter)*printIter && Iter>1
        tid=round(toc(tstart)/(Iter-1)*(nIter-Iter+1));
        fprintf('Time spent: %g. Starting rep no.: %g Estimated time left until %g reps: %g seconds. \n'...
            ,toc(tstart),Iter,nIter,tid);
        if Iter > 2
            fprintf('Relative change in lower bound %d  \n', abs(deltaL)/abs(Lbound(Iter-1)));
        end
    end
    
    %% Z 
    Prec_z = eye(K);
    for m=1:M
        Prec_z = Prec_z + A(:,:,m)'*Epsi(:,:,m)*A(:,:,m);
        for d=1:D
            Prec_z = Prec_z + Epsi(d,d,m)*Sig_ad(:,:,d,m); % Ændres til *diag(diag(Sig_wd(:,:,d,m))) ?
        end
    end
    Sig_z = inv(Prec_z);
    
    for m=1:M
        Zspec(:,:,m) = A(:,:,m)'*Epsi(:,:,m)*X(:,:,m);
    end
    Z = Sig_z*sum(Zspec,3);
    Ezz = N*Sig_z + Z*Z'; 
    
    ZX=zeros(K,D,M);
    for m=1:M
        ZX(:,:,m) = Z*X(:,:,m)';
    end
    
    
    %% A
    Aold=A;
    for m=1:M
        Tr_sigak = zeros(K,1); % Trace(Sig_ak)
        for d=1:D
            Sig_ad(:,:,d,m) = inv(Epsi(d,d,m)*Ezz + diag(Elambda(:,m)));
            
            sum1 = Ezz*Aold(1:end~=d,:,m)'*Epsi(d,1:end~=d,m)';
            
            sum2 = ZX(:,:,m)*Epsi(d,:,m)';
            
            A(d,:,m) = Sig_ad(:,:,d,m) * (-sum1 + sum2 + Elambda(:,m).*U(d,:)');
            Tr_sigak = Tr_sigak + diag(Sig_ad(:,:,d,m));
        end
        
        Eaa(:,m) = Tr_sigak + sum(A(:,:,m).^2)';
    end
    
    
    %% U
    sig_u = (sum(Elambda,2) + Ealpha).^(-1);
    lambda_rep = repmat(Elambda,[1,1,D]);
    lambda_shift = shiftdim(lambda_rep,2);
    
    for k=1:K
        U(:,k) = sig_u(k)*sum(lambda_shift(:,k,:).*A(:,k,:),3);
    end
    Tr_sigu = sig_u*D;
    Euu = Tr_sigu + sum(U.^2,1)';
    
    %% Alpha
    b_alpha = b0 + Euu/2;
    Ealpha = a_alpha./b_alpha;
    
    
    %% lambda
    b_l = repmat(Euu/2,[1,M]) + Eaa/2 - permute(sum(A.*repmat(U,[1,1,M]),1),[2,3,1]);
    if strcmp(lSize,'KM')
        b_l2 = b_l;
    elseif strcmp(lSize,'K')
        b_l2 = repmat(sum(b_l,2),[1,M]);
    else
        b_l2 = sum(sum(b_l)) * ones(K,M);
    end
    
    b_lambda = b0 + b_l2;
    Elambda = a_lambda./b_lambda;
    
    
    %% Psi
    for m=1:M
        cov_apa = zeros(D,D);
        for d=1:D
            cov_apa(d,d) = trace(Sig_ad(:,:,d,m)*Ezz);
        end
        Epsi(:,:,m) = v_psi*eye(D) / (A(:,:,m)*Ezz*A(:,:,m)' + cov_apa ...
            + XX(:,:,m) - ZX(:,:,m)'*A(:,:,m)' - (ZX(:,:,m)'*A(:,:,m)')' + S0inv(:,:,m));
    end
    
    
    %% L(q) (Lower bound)
    Lq = 0;
    for m=1:M
        Lq = Lq + v_psi*log(det(Epsi(:,:,m)));
        for d=1:D
            Lq = Lq + log(det(Sig_ad(:,:,d,m)));
        end
    end
    Lq = Lq/2;
    
    Lq = Lq - sum(a_alpha*log(b_alpha)) + sum(log(sig_u))*D/2;
    
    L_zz = sum(sum(Z.^2));
    
    if strcmp(lSize,'KM')
        L_lambda = -a_lambda*sum(sum(log(b_lambda)));
    elseif strcmp(lSize,'K')
        L_lambda = -a_lambda*sum(log(b_lambda(:,1)));
    else
        L_lambda = -a_lambda*log(b_lambda(1,1));
    end
    
    
    Lq = Lq + L_lambda + N*log(det(Sig_z))/2 - (N*trace(Sig_z) + L_zz)/2;
    
    Lbound(Iter) = Lq;
    
    
    %% Konvergering
    if Iter > 1
        deltaL = Lbound(Iter)-Lbound(Iter-1);
        if abs(deltaL/Lbound(Iter)) < threshold
            if verbose
                disp(['Converged in ' num2str(Iter) ' iterations'])
            end
            run=0;
        end
    end
    
    if Iter==nIter
        run = 0;
    end
    
    if run==1
        Iter = Iter + 1;
    end
    
end
if Iter == nIter
    if verbose
        disp(['Not converged within ' num2str(nIter) ' iterations'])
    end
    converged = 0;
    
end

if verbose
    fprintf('Time spent: %g\n',toc(tstart))
end
end
