function [theta_jplus1,e_j] = ILCBF(POI,grids,Ts,N_trial,theta0,r,Psi,t,a,b,x0)
if ishandle(1)
    close(1);
end
%% inputs
N = length(t);
Tend = t(end);
dirs = size(r,2);


%% system
G = ss(ModelFlexiblePlateFirstPrinciple(POI,grids,a,b)); % model for simulation

load controller1
sys(1,1) = c2d(shapeit_data.P.sys,Ts); % model for ILCBF
controller(1,1) = d2d(ss(shapeit_data.C_tf_z),Ts);
load controller2
sys(2,2) = c2d(shapeit_data.P.sys,Ts); % model for ILCBF
controller(2,2) = d2d(ss(shapeit_data.C_tf_z),Ts);
load controller3
sys(3,3) = c2d(shapeit_data.P.sys,Ts); % model for ILCBF
controller(3,3) = d2d(ss(shapeit_data.C_tf_z),Ts);

PS = feedback(sys,controller,-1);
%% weighting
We          = eye(dirs*N)*1e6;
Wf          = eye(dirs*N)*0e3;
WDf         = eye(dirs*N)*0e-1;
%% BF

npsi = size(Psi,2);
JPsi = zeros(dirs*N,npsi);

for iBF = 1:npsi
    JPsi(:,iBF) = reshape(lsim(PS,reshape(Psi(:,iBF),N,dirs)),dirs*N,1);
end
R = JPsi.'*We*JPsi+Psi.'*(Wf+WDf)*Psi;
Rinv = eye(size(R,2))/R;

Q = Rinv*(JPsi.'*We*JPsi+Psi.'*WDf*Psi);
L = Rinv*(JPsi.'*We);

%% init ILC with BF
theta_jplus1 = theta0;
f_jplus1 = reshape(Psi*theta_jplus1,N,dirs);
%% initialize storage andplotting for ILC

% Initialize storage variables.
history.f           = NaN(N,dirs,N_trial);
history.u           = NaN(N,dirs,N_trial);
history.e           = NaN(N,dirs,N_trial);
history.eNorm       = NaN(dirs,N_trial);
history.eInfNorm    = NaN(dirs,N_trial);
history.theta_j     = NaN(npsi,N_trial);

PlotTrialDataMIMO;

for trial = 1:N_trial
    f_j = f_jplus1;
    theta_j(:,trial) = theta_jplus1;
    out = sim('flexibleBeamILCBF','SrcWorkspace','current');
    
    % load simulation data:
    u_j = out.simout(:,1:dirs);
    
    e_j = out.simout(:,dirs+1:end);
    
    % Store trial data.
    history.f(:,:,trial)          = f_j;
    history.u(:,:,trial)          = u_j;
    history.e(:,:,trial)          = e_j;
    for i = 1:dirs
        history.eNorm(i,trial)        = norm(e_j(:,i),2);
        history.eInfNorm(i,trial)     = norm(e_j(:,i),Inf);
    end
    history.theta_j(:,trial)      = theta_j(:,trial);
    
    PlotTrialDataMIMO;
    
    theta_jplus1 = (Q*theta_j(:,trial)+L*reshape(e_j,dirs*N,[]));
    f_jplus1 = reshape(Psi*theta_jplus1,N,dirs);
end
if false
   figure(2);clf;
   subplot(121)
   semilogy(history.eNorm,'s--','Markersize',12,'Linewidth',1.3); hold on;
   xlabel('Trial Number [-]'); ylabel('$\|e\|_2$ [$m$]');
   ylim([1.5e-5 7.5e-4]); xticks(1:6); xlim([0.7 6.3]);
   subplot(122)
   yyaxis left
   plot(theta_j(1,:),'^--','Markersize',12,'Linewidth',1.3);
   ylabel('Acceleration Parameter [$kg$]');
   xlabel('Trial Number [-]');
   ylim([-0.005 0.165])
   yyaxis right
   plot(theta_j(2,:),'o--','Markersize',12,'Linewidth',1.3);
   ylabel('Snap Parameter [$kg/s^2$]'); 
   xticks(1:6); xlim([0.7 6.3]); ylim([-0.1e-5 3.5e-5])
end
end

