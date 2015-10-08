function [W,L]=corrca(X1,X2)
[D,T] = size(X1); 

[Wtemp,Ltemp] = eig((X1*X2'+X2*X1')/T , (X1*X1'+X2*X2')/T);

Lvec = diag(Ltemp);
[B,I] = sort(abs(Lvec),'descend');
L = diag(Lvec(I));
W=Wtemp(:,I);
end