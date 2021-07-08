function [ystar,deltay] = OptimizeMI2D(xTest,xTraining,hyps,covfunc)
%% define/initialize
Ntest = size(xTest,1);
deltay = zeros(1,Ntest);
A = unique(xTraining,'rows'); 
delta = 1e-6;
%% calculations

KAA = feval(covfunc{:},hyps.cov,xTraining);
KAAplus = KAA+delta*mean(mean(KAA))*eye(length(A));

for j = 1:Ntest
    Ab = setdiff(xTest,union(A,xTest(j,:),'rows'),'rows'); 
    
    KyAb = feval(covfunc{:},hyps.cov,xTest(j,:),Ab);
    KyA = feval(covfunc{:},hyps.cov,xTest(j,:),A);
    Kyy = feval(covfunc{:},hyps.cov,xTest(j,:));
    
    KAbAb = feval(covfunc{:},hyps.cov,Ab);
    KAbAbplus = KAbAb+delta*mean(mean(KAbAb))*eye(length(Ab));

    deltay(j) = (Kyy-KyA/KAAplus*KyA')/(Kyy-KyAb/KAbAbplus*KyAb');
end
deltay(deltay==Inf)=min(min(deltay));
[~,I] = max(deltay);
%% result
ystar = xTest(I,:); 
end

