close all; clear all; clc;
SetPlotLatexStyle;
opts = DefBodeOpts;
[c1,c2,c3,c4,c5,c6,c7] = MatlabDefaultPlotColors;
%%
grids = 17; % square gridded (grids * grids) (so grid sizes can be non-equidistant)
Lx = 0.25;  % [m]
Ly = 0.25;  % [m]

C = zeros(grids,grids,1);
C(ceil(grids/2),ceil(grids/2),1) = 1;     % center
% C(2,2,2) = 1;       % left upper corner
% C(end-1,end-1,3) = 1;   % right upper corner
% C(end-1,2,4) = 1;   % left bottom corner
% C(2,end-1,5) = 1;   % right bottom corner

n = size(C,3);  % [-] amount of ini training positions 
n2 = n+6;       % [-] amount of total training positions

for i = 1:n
    [row,col] = find(C(:,:,i));
    xTraining(i,:) = [-Lx+2*(col-1)/(grids-1)*Lx -Ly+2*(row-1)/(grids-1)*Ly];
end

xpv = linspace(-Lx,Lx,grids);
ypv = linspace(-Ly,Ly,grids);
[xv, yv] = meshgrid(xpv,ypv);
xTest = [xv(:) yv(:)];

%%
Ts = 1e-3;
N_trial = 8;
[ty,ddy] = make4(5e-4,1e-3,1e-2,2.5e-1,2e1,Ts); % good choice: 5e-4,1e-3,1e-2,2.5e-1,2e1
[~,t,s,j,a,v,r1,~] = profile4(ty,ddy(1),Ts);
Psi = [v a j s];
[ty,ddy] = make4(1e-1,5e-1,5e0,1e2,5e3,Ts); % good choice: 5e-4,1e-3,1e-2,2.5e-1,2e1
[~,~,s,j,a,v,r2,~] = profile4(ty,ddy(1),Ts);
Nd = size(Psi,1)-length(r2);
Psi = blkdiag(Psi,[a;zeros(Nd,1)],[a;zeros(Nd,1)]);
npsi = size(Psi,2);
theta0 = zeros(npsi,1);


for i = 1:n
    r = [r1 [r2;r2(end)*ones(Nd,1)]+xTraining(i,1) [r2;r2(end)*ones(Nd,1)]+xTraining(i,2)];
    [theta(:,i), e(:,:,i)] = ILCBF(squeeze(C(:,:,i)),grids,Ts,N_trial,theta0,r,Psi,t,Lx,Ly);
end
%% GP
meanfunc = {@meanConst};
% covfunc = {@covSEard};
covfunc = {@covProd,{{@covSEiso},{@covSEiso}}};
likfunc = {@likGauss};


hypGuess.cov = log([1e2 sqrt(1e-3) 1e2 sqrt(1e-3);
                    5e0 sqrt(5e-2)  5e0 sqrt(5e-2);
                    1e2 sqrt(1e-3) 1e2 sqrt(1e-3);
                    5e0 sqrt(1e-5) 5e0 sqrt(1e-5);
                    5e0 sqrt(5e-2)  5e0 sqrt(5e-2);
                    5e0 sqrt(5e-2)  5e0 sqrt(5e-2)
                    ]);
hypGuess.lik = log(1e-6*min(abs(theta),[],2));
hypGuess.mean = mean(theta,2);

hypOpt.cov = hypGuess.cov(npsi-2,:);
hypOpt.lik = hypGuess.lik(npsi-2);
hypOpt.mean = hypGuess.mean(npsi-2);

infMethod = @infVB;

for i = n+1:n2
    Y = theta(npsi-2,:)';
    
    hypOpt = minimize(hypOpt, @gp, -500, infMethod, meanfunc, covfunc, likfunc, xTraining, Y);
    [mu, s2] = gp(hypOpt, infMethod, meanfunc, covfunc, likfunc, xTraining, Y, xTest);
    figure(2); clf;
    subplot(121)
    surf(xpv,ypv,reshape(mu,grids,[]))
    hold on
    plot3(xTraining(:,1),xTraining(:,2),Y,'^','MarkerSize',15,'MarkerFaceColor',c2,'MarkerEdgeColor',c2);
    xlabel('x [m]');ylabel('y [m]');zlabel('Snap Parameter [$kg/s^2$]')
    
    
    [xstar,deltay] = OptimizeMI2D(xTest,xTraining,hypOpt,covfunc);
    subplot(122)
    surf(xpv,ypv,reshape(deltay,grids,[])); hold on;
    plot(xTraining(:,1),xTraining(:,2),'o','MarkerSize',15,'MarkerFaceColor',c2,'MarkerEdgeColor',c2)
    xlabel('x [m]');ylabel('y [m]'); zlabel('$\delta$ in mutual information');
    zlim([0 3e5]); set(gca,'zscale','log');
    xTraining(i,:) = xstar;
    col = round((xstar(1)+Lx)/(2*Lx)*(grids-1)+1);
    row = round((xstar(2)+Ly)/(2*Ly)*(grids-1)+1);
    C(row,col,i) = 1;
    r = [r1 [r2;r2(end)*ones(Nd,1)]+xTraining(i,1) [r2;r2(end)*ones(Nd,1)]+xTraining(i,2)];
    [theta(:,i), e(:,:,i)] = ILCBF(squeeze(C(:,:,i)),grids,Ts,N_trial,theta0,r,Psi,t,Lx,Ly);
end

%% model all ff parameters as function of position
hypOpt(npsi-2,:) = hypOpt;
for i = [1:npsi-3 npsi-2:npsi]
    hypOpt(i,:).cov = hypGuess.cov(i,:);
    hypOpt(i).lik = hypGuess.lik(i);
    hypOpt(i).mean = hypGuess.mean(i);
end
figure(3); clf;

for i = 1:npsi
    Y = theta(i,:)';
    hypOpt(i,:) = minimize(hypOpt(i,:), @gp, -500, infMethod, meanfunc, covfunc, likfunc, xTraining, Y);
    [mu(:,i), ~] = gp(hypOpt(i,:), infMethod, meanfunc, covfunc, likfunc, xTraining, Y, xTest);
    subplot(2,3,i);
    surf(xpv,ypv,reshape(mu(:,i),grids,[]))
    hold on
    plot3(xTraining(:,1),xTraining(:,2),Y,'^','MarkerSize',15,'MarkerFaceColor',c2,'MarkerEdgeColor',c2);
    xlabel('x [m]');ylabel('y [m]');zlabel('Snap Parameter [$kg/s^2$]');
end

%% testing
Ntest = 25;
xiEval = round(linspace(1,grids,5));
yiEval = round(linspace(1,grids,5));
[iEvalx,iEvaly] = meshgrid(xiEval,yiEval);
iEval = [iEvalx(:) iEvaly(:)]; 
xEval = -Lx+2*(iEval-1)./(grids-1).*[Lx Ly];
Ctest = zeros(grids,grids,Ntest);

for i = 1:npsi
    [thetaTest(i,:), ~] = gp(hypOpt(i,:), @infVB, meanfunc, covfunc, likfunc, xTraining, theta(i,:)', xEval);
end
% thetaTest(1:end-1,:) = repmat(theta(1:end-1,1),1,Ntest);

for i = 1:Ntest
    Ctest(iEval(1),iEval(2),i) = 1;
    [~, eGP(:,i)] = ILCBF(squeeze(Ctest(:,:,i)),grids,Ts,1,thetaTest(:,i),r,Psi,t,Lx,Ly);
    eNormGP(i) = norm(eGP(:,i),2);
    [~, eConstant(:,i)] = ILCBF(squeeze(Ctest(:,:,i)),grids,Ts,1,theta(:,1),r,Psi,t,Lx,Ly);
    eNormConstant(i) = norm(eConstant(:,i),2);
end
%% visualization
figure(4);clf;
surf(-Lx+2*(xiEval-1)./(grids-1).*Lx,-Ly+2*(yiEval-1)./(grids-1).*Ly,reshape(eNormGP,5,[]));
hold on
surf(-Lx+2*(xiEval-1)./(grids-1).*Lx,-Ly+2*(yiEval-1)./(grids-1).*Ly,reshape(eNormConstant,5,[]));
set(gca,'Zscale','log');
xlabel('Scheduling Variable $x$ [$m$]');
xlabel('Scheduling Variable $y$ [$m$]');
zlabel('$\|e\|_2$ [$m$]');
legend('GP Snap Feefdorward','Position-Independent Feedforward');

figure(5);clf;
plot(t,eGP(:,1));
hold on
plot(t,eConstant(:,1));