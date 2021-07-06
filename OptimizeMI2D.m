function [ystar,deltay] = OptimizeMI2D(xTest,xTraining,hyps,covfunc)
%% define/initialize
Ntest = size(xTest,1);
deltay = zeros(1,Ntest);
A = unique(xTraining,'rows');
Na = size(A,1);
%% calculations

KAA = feval(covfunc{:},hyps.cov,xTraining);

for j = 1:Ntest
    Ab = setdiff(xTest,union(A,xTest(j,:),'rows'),'rows');
    
    KAbAb = feval(covfunc{:},hyps.cov,Ab);
    KyAb = feval(covfunc{:},hyps.cov,xTest(j,:),Ab);
    KyA = feval(covfunc{:},hyps.cov,xTest(j,:),A);
    Kyy = feval(covfunc{:},hyps.cov,xTest(j,:));
    
%     invKAbAb = ((eye(length(Ab))))/(((KAbAb+hyps(3,GPI)*eye(length(Ab)))));
    
%     HA = h(A(:,1),A(:,2));
%     HAb = h(Ab(:,1),Ab(:,2));
    KAbAbplus = KAbAb+1e-14*eye(length(Ab));

    deltay(j) = (Kyy-KyA/KAA*KyA')/(Kyy-KyAb/KAbAbplus*KyAb');
end
deltay(deltay==Inf)=min(min(deltay));
[~,I] = max(deltay);
%% result
ystar = xTest(I,:); 
end

