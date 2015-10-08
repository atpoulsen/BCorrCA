function [X,Z,Atrue,Utrue,Y] = synthData(D,N,M,K,lambda,snr,noisetype, sourcetype)
%% Set parameters
if ~exist('lambda','var')
    lambda = 1;
end
if ~exist('N','var')
    N = 500;
end
if ~exist('D','var')
    D = 6;
end
if ~exist('K','var')
    K = 4;
end
if ~exist('M','var')
    M = 5;
end
if ~exist('noisetype','var')
    noisetype = 'average';
end
if ~exist('sourcetype','var')
    sourcetype = 'klami';
end

if strcmp(sourcetype,'klami')
    Z = zeros(N,K);
    Z(:,1) = sin(linspace(0,4*pi,N));
    Pz1 = mean(Z(:,1).^2);
    Z(:,1) = Z(:,1)/sqrt(Pz1);
    if K >= 2
        Z(:,2) = cos(linspace(0,4*pi,N));
        Pz2 = mean(Z(:,2).^2);
        Z(:,2) = Z(:,2)/sqrt(Pz2);
    end
    if K >= 3
        Z(:,3) = linspace(-1,1,N);
        Pz3 = mean(Z(:,3).^2);
        Z(:,3) = Z(:,3)/sqrt(Pz3);
    end
    if K >= 4
        Z(:,4) = linspace(-1,1,N).^2;
        Z(:,4) = Z(:,4) - mean(Z(:,4));
        Pz4 = mean(Z(:,4).^2);
        Z(:,4) = Z(:,4)/sqrt(Pz4);
    end
    
else
    Z = randn(N,K);
    Pz = mean(Z.^2);
    for k=1:K
        Z(:,k) = Z(:,k)/sqrt(Pz(k));
    end
end

alpha = 1;
%% Generate Data
Y = struct('data',{},'W',{});

Utrue = randn(D,K);

for m = 1:M
    Y(m).W = Utrue;
end

X = zeros(D,N,M);
Atrue = zeros(D,K,M);
for view = 1:M
    %% Mixing sources
    % Disable sources
    for k = 1:K
        Y(view).W(:,k) = (Y(view).W(:,k)./sqrt(alpha));
    end
    
    % Separate weights
    Y(view).W = Y(view).W + randn(D,K)./sqrt(lambda);
    
    % Mixing sources
    Y(view).data = Z * Y(view).W';
    
    %% Add observation noise.
    if strcmp(noisetype,'channel')
        % Channel specific variance based on channel power and requested SNR.
        for d=1:D
            P = mean(Y(view).data(:,d).^2);
            sig_n = sqrt(P*10^(-snr/10));  % Noise std, for each channel in Z*W'
            Y(view).data(:,d) = Y(view).data(:,d) + randn(N,1)*sig_n;
        end
        
    elseif strcmp(noisetype,'average')
        % Variance based on average channel power and requested SNR.
        P = mean(mean(Y(view).data.^2));
        sig_n = sqrt(P*10^(-snr/10));  % Noise std, tilpasset hver kanal i Z*W'
        Y(view).data = Y(view).data + randn(N,D)*sig_n;
    else
        error('Choose a noise type')
    end
    
    %% Normalize data to have a average power of 1
    Px = mean(mean(Y(view).data.^2));
    Y(view).data = Y(view).data/sqrt(Px);
    
    %% Copy data to other structures
    X(:,:,view)=Y(view).data';
    Atrue(:,:,view) = Y(view).W;
end