function [ystar,deltay] = OptimizeMI2D(V,A,hyps,covfunc)
%% define/initialize
Ntest = size(V,1);
deltay = zeros(1,Ntest);
delta = 1e-6;
%% calculations

KAA = feval(covfunc{:},hyps.cov,A);
KAAplus = KAA+delta*mean(mean(KAA))*eye(length(KAA));

for j = 1:Ntest
    Ab = setdiff(V,union(A,V(j,:),'rows','stable'),'rows','stable'); 
    
    KyAb = feval(covfunc{:},hyps.cov,V(j,:),Ab);
    KyA = feval(covfunc{:},hyps.cov,V(j,:),A);
    Kyy = feval(covfunc{:},hyps.cov,V(j,:));
    
    KAbAb = feval(covfunc{:},hyps.cov,Ab);
    KAbAbplus = KAbAb+delta*mean(mean(KAbAb))*eye(length(KAbAb));

    deltay(j) = (Kyy-KyA/KAAplus*KyA')/(Kyy-KyAb/KAbAbplus*KyAb');
end
deltay(deltay==Inf)=min(min(deltay));
[~,I] = max(deltay);
%% result
ystar = V(I,:); 
end

