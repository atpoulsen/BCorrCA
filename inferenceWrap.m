function [iZ,W,lambda,psi,alpha,Lbound,U] = inferenceWrap( algo, X)
% Predefine  variables
W=[];
lambda=0;
psi=[];
alpha=[];
Lbound = [];
U=[];
[D,N,M]=size(X);

switch algo
    case 'CorrCA'
        if M>2
            Npaired = N*M*(M-1)/2;
            Xpaired = zeros(D,Npaired,2);
            ip=0;
            for m=1:M
                for mm=m+1:M
                    n = 1+ip*N : N*(1+ip);
                    Xpaired(:,n,1)=X(:,:,m);
                    Xpaired(:,n,2)=X(:,:,mm);
                    ip = ip+1;
                end
            end
            [W,~]=corrca(Xpaired(:,:,1), Xpaired(:,:,2));
            
            iZ=zeros(N,D);
            for m = 1:M
                iZ = iZ + X(:,:,m)'*W; % sum of dataset timeseries
            end
            iZ = iZ./M;
            U = W;
            
        else
            [W,~]=coca(X(:,:,1), X(:,:,2));
            iZ = (X(:,:,1)'*W + X(:,:,2)'*W)./2; % sum of dataset timeseries
            U = W;
        end
    case 'CCA'
        if M>2
            Npaired = N*M*(M-1)/2;
            Xpaired = zeros(D,Npaired,2);
            ip=0;
            for m=1:M
                for mm=m+1:M
                    n = 1+ip*N : N*(1+ip);
                    Xpaired(:,n,1)=X(:,:,m);
                    Xpaired(:,n,2)=X(:,:,mm);
                    ip = ip+1;
                end
            end
            [W1,W2,~,Z1,Z2] = canoncorr(Xpaired(:,:,1)', Xpaired(:,:,2)');
            
            iZ=zeros(N,D);
            for i = 0 : M*(M-1)/2-1
                n = 1+i*N : N*(1+i);
                iZ = iZ + (Z1(n,:) + Z2(n,:))./2; % sum of dataset timeseries
            end
            iZ = iZ./ (M*(M-1)/2);
            W(:,:,1) = W1;
            W(:,:,2) = W2;
            U = mean(W,3);
            
        else
            [W1,W2,~,Z1,Z2] = canoncorr(X(:,:,1)',X(:,:,2)');
            iZ = (Z1 + Z2)./2;
            
            W(:,:,1) = W1;
            W(:,:,2) = W2;
            U = mean(W,3);
        end
    otherwise
        disp('*         Wrong input to inferenceWrap          *')
        
end