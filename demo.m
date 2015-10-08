clear; close all;clc
%% Data info
snr_range = -20:5:10;
M = 5; % views
K0 = 1; % no. of hidden sources
N = 500; % samples per view
D = 8; % input dimension
lambda=1; % similarity with shared forward model, U.
noisetype = 'average'; % 'average': noise normalised within each view. 'channel': for each channel.
sourcetype = 'klami'; % 'klami': the four sources used in Klami (2013), otherwise gaussian.

%% Simulation settings
reps = 10;

opts.verbose=0;
opts.K = 1;

%%
ind = 0;
disp('Running simulation for SNR:')
for snr = snr_range
    disp(snr)
    ind = ind + 1;
    for r = 1:reps
        % creating random data
        [X,Z] = synthData(D,N,M,K0,lambda,snr,noisetype,sourcetype);
        
        [~,iZ] = BCorrCA(X,opts);
        cc(1,r) = calcCorr(iZ', Z);
        
        iZ = inferenceWrap('CorrCA', X);
        cc(2,r) = calcCorr(iZ, Z);
        iZ = inferenceWrap('CCA', X);
        cc(3,r) = calcCorr(iZ, Z);
    end
    cc_mean(:,ind) = mean(cc,2);
    cc_sem(:,ind) = std(cc,0,2)/sqrt(reps);
end

figure
mseb(snr_range,cc_mean,cc_sem,[],1);
xlabel('SNR'),ylabel('Mean Correlation')
legend('BCorrCA','CorrCA','CCA')