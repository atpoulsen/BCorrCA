function [cor_max, corcomb] = calcCorr(iZ, Z)
% Finds all combinations between true sources and the estimated ones in a
% manner where the same source is not correlated twice.
% Ktrue <= K must be true. For K>=10, the calculations become >3.5 seks

% if size(iZ,2) > 6
%     iZ=iZ(:,1:6);
% end

Ktrue = size(Z,2);
K = size(iZ,2);

% All correlations
signalCorrelation = corr(iZ,Z);
signalCorrelation( isnan(signalCorrelation) )=0;


% All combinations
perm_dummy = perms(1:K);
perm_rows = 1:factorial(K-Ktrue):size(perm_dummy,1);
perm = perm_dummy(perm_rows,1:Ktrue);
cor_max = 0;
corcomb = 1:Ktrue;
for i = 1:size(perm,1)
    cors = diag(signalCorrelation(perm(i,:),:));
    cor_avg = mean(abs(cors));
    if cor_avg > cor_max
        cor_max = cor_avg;
        corcomb = perm(i,:);
    end
end

end
