close all; clear all; clc;
SetPlotLatexStyle;
opts = DefBodeOpts;
[c1,c2,c3,c4,c5,c6,c7] = MatlabDefaultPlotColors;
%%
grids = 31; % square gridded (grids * grids) (so grid sizes can be non-equidistant)
Lx = 0.25;  % [m]
Ly = 0.25;  % [m]
n=5;        % [-] amount of training positions

C = zeros(grids,grids,n);
C(16,16,1) = 1;     % center
C(3,3,2) = 1;       % left upper corner
C(3,end-2,3) = 1;   % right upper corner
C(end-2,3,4) = 1;   % left bottom corner
C(end-2,end-2,5) = 1;   % right bottom corner

for i = 1:size(C,3)
    [row,col] = find(C(:,:,i));
    xTraining(i,:) = [-Lx+2*(col-1)/(grids-1)*Lx -Ly+2*(row-1)/(grids-1)*Ly];
end

xpv = -Lx:0.025:Lx;
ypv = -Ly:0.025:Ly;
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

Y = theta(end,:)';
hypGuess = struct('mean',[], 'cov', [log(2e1) log(2e1) log(mean(abs(Y)))], 'lik', log(1e-100));
hypOpt = minimize(hypGuess, @gp, -500, @infGaussLik, meanfunc, covfunc, likfunc, xTraining, Y); % optimize hyperparameters
[mu, s2] = gp(hypOpt, @infGaussLik, meanfunc, covfunc, likfunc, xTraining, Y, xTest);


figure(2); clf;
surf(xpv,ypv,reshape(mu,21,[]))
hold on
plot3(xTraining(:,1),xTraining(:,2),Y,'^','MarkerSize',15,'MarkerFaceColor',c2,'MarkerEdgeColor',c2);

Kss = feval(covfunc{:},hypOpt.cov,[xTest(:,1) xTest(:,1)]);
Ks = feval(covfunc{:},hypOpt.cov,xTraining,xTest);
[ystar,deltay] = OptimizeMI2D(xTest,xTraining,hypOpt,covfunc);
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