close all; clear all; clc;
SetPlotLatexStyle;
opts = DefBodeOpts;
[c1,c2,c3,c4,c5,c6,c7] = MatlabDefaultPlotColors;
%%
grids = 19; % square gridded (grids * grids) (so grid sizes can be non-equidistant)
Lx = 0.25;  % [m]
Ly = 0.25;  % [m]
n=5;        % [-] amount of ini training positions
n2 = 12;     % [-] amount of total training positions

C = zeros(grids,grids,n);
C(ceil(grids/2),ceil(grids/2),1) = 1;     % center
C(2,2,2) = 1;       % left upper corner
C(end-1,end-1,3) = 1;   % right upper corner
C(end-1,2,4) = 1;   % left bottom corner
C(2,end-1,5) = 1;   % right bottom corner

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
[~,t,s,j,a,v,r,~] = profile4(ty,ddy(1),Ts);
Psi = [v a j s];
npsi = size(Psi,2);
theta0 = zeros(npsi,1);


for i = 1:n
    [theta(:,i), e(:,i)] = ILCBF(squeeze(C(:,:,i)),grids,Ts,N_trial,theta0,r,Psi,t,Lx,Ly);
end
%% GP
meanfunc = {@meanZero};
covfunc = {@covSEard};
likfunc = {@likGauss};
hypOpt = struct('mean',[], 'cov', [log(1e1) log(1e1) log(mean(abs(theta(end,:))))], 'lik', log(1e-4*min(abs(theta(end,:)))));'lik', log(1e-10));

for i = n+1:n2
    Y = theta(end,:)';

    hypOpt = minimize(hypOpt, @gp, -500, @infVB, meanfunc, covfunc, likfunc, xTraining, Y);
    [mu, s2] = gp(hypOpt, @infVB, meanfunc, covfunc, likfunc, xTraining, Y, xTest);
    figure(2); clf;
    subplot(121)
    surf(xpv,ypv,reshape(mu,grids,[]))
    hold on
    Y = theta(end,:)';
    plot3(xTraining(:,1),xTraining(:,2),Y,'^','MarkerSize',15,'MarkerFaceColor',c2,'MarkerEdgeColor',c2);
    xlabel('x');
    ylabel('y');
    
    
    [xstar,deltay] = OptimizeMI2D(xTest,xTraining,hypOpt,covfunc);
    subplot(122)
    surf(xpv,ypv,reshape(deltay,grids,[])); hold on;
    plot(xTraining(:,1),xTraining(:,2),'o','MarkerSize',15,'MarkerFaceColor',c2,'MarkerEdgeColor',c2)
    xlabel('x');
    ylabel('y');
    xTraining(i,:) = xstar;
    col = round((xstar(1)+Lx)/(2*Lx)*(grids-1)+1);
    row = round((xstar(2)+Ly)/(2*Ly)*(grids-1)+1);
    C(row,col,i) = 1;
    [theta(:,i), e(:,i)] = ILCBF(squeeze(C(:,:,i)),grids,Ts,N_trial,theta0,r,Psi,t,Lx,Ly);
end


figure(2); clf;
surf(xpv,ypv,reshape(mu,grids,[]))
hold on
Y = theta(end,:)';
plot3(xTraining(:,1),xTraining(:,2),Y,'^','MarkerSize',15,'MarkerFaceColor',c2,'MarkerEdgeColor',c2);
%% testing
Ntest = 10;
iEval = randi(grids,Ntest,2);% some random indices
xEval = -Lx+2*(iEval-1)./(grids-1).*[Lx Ly];
Ctest = zeros(grids,grids,Ntest);

[thetaSnapTest, ~] = gp(hypOpt, @infGaussLik, meanfunc, covfunc, likfunc, xTraining, Y, xEval);
thetaTest = [repmat(theta(1:end-1,1),1,Ntest);thetaSnapTest'];

for i = 1:Ntest
    Ctest(iEval(1),iEval(2),i) = 1;
    [~, eGP(:,i)] = ILCBF(squeeze(Ctest(:,:,i)),grids,Ts,1,thetaTest(:,i),r,Psi,t,Lx,Ly);
    eNormGP(i) = norm(eGP(:,i),2);
    [~, eConstant(:,i)] = ILCBF(squeeze(Ctest(:,:,i)),grids,Ts,1,theta(:,1),r,Psi,t,Lx,Ly);
    eNormConstant(i) = norm(eConstant(:,i),2);
end
%% visualization
figure(3);clf;
plot3(xEval(:,1),xEval(:,2),eNormGP,'s','Markersize',15,'Linewidth',1.3);
hold on
plot3(xEval(:,1),xEval(:,2),eNormConstant,'^','Markersize',15,'Linewidth',1.3);
set(gca,'Zscale','log');
xlabel('Scheduling Variable $\rho_1$ [$m$]');
xlabel('Scheduling Variable $\rho_2$ [$m$]');
zlabel('$\|e\|_2$ [$m$]');
legend('GP Snap Feefdorward','Position-Independent Feedforward');

figure(4);clf;
plot(t,eGP(:,1));
hold on
plot(t,eConstant(:,1));