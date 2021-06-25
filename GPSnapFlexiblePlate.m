close all; clear all; clc;
SetPlotLatexStyle;
opts = DefBodeOpts;
[c1,c2,c3,c4,c5,c6,c7] = MatlabDefaultPlotColors;
%%
grids = 31; % square gridded (grids * grids) (so grid sizes can be non-equidistant)
Lx = 0.25;  % [m]
Ly = 0.25;  % [m]
n=5; % [-] amount of training positions

C = zeros(grids,grids,n);
C(16,16,1) = 1; % center
C(3,3,2) = 1; % left upper corner
C(3,end-2,3) = 1; % right upper corner
C(end-2,3,4) = 1; % left bottom corner
C(end-2,end-2,5) = 1; % right bottom corner

for i = 1:size(C,3)
    [row,col] = find(C(:,:,i));
    xTraining(i,:) = [-Lx+2*col/grids*Lx -Ly+2*row/grids*Ly];
end

xpv = -Lx:0.025:Lx;
ypv = -Ly:0.025:Ly;
[xv, yv] = meshgrid(xpv,ypv);
xTest = [xv(:) yv(:)];


%%
Ts = 1e-3;
N_trial = 16;
[ty,ddy] = make4(5e-4,1e-3,1e-2,2.5e-1,2e1,Ts); % good choice: 5e-4,1e-3,1e-2,2.5e-1,2e1
[~,t,s,j,a,v,r,~] = profile4(ty,ddy(1),Ts);
Psi = [a s];
npsi = size(Psi,2);
theta0 = zeros(npsi,1);


for i = 1:n
    [theta(:,i) e(:,i)] = ILCBF(squeeze(C(:,:,i)),grids,Ts,N_trial,theta0,r,Psi,t,Lx,Ly);
end
%%
meanfunc = {@meanZero};
covfunc = {@covSEard};
likfunc = {@likGauss};

Y = theta(2,:)';
hypGuess = struct('mean',[], 'cov', [3e0 3e0 log(mean(abs(Y)))], 'lik', log(1e-9));
hypOpt = minimize(hypGuess, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, xTraining, Y); % optimize hyperparameters
[mu, s2] = gp(hypOpt, @infGaussLik, meanfunc, covfunc, likfunc, xTraining, Y, xTest);


figure(2); clf;
surf(xpv,ypv,reshape(mu,21,[]))
hold on
plot3(xTraining(:,1),xTraining(:,2),Y,'^','MarkerSize',15,'MarkerFaceColor',c2,'MarkerEdgeColor',c2)
