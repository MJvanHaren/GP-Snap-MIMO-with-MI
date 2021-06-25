function [theta_jplus1,e_j] = ILCBF(POI,grids,Ts,N_trial,theta0,r,Psi,t,a,b)
if ishandle(1)
    close(1);
end
%% inputs
N = length(t);
Tend = t(end);


%% system
G = ss(ModelFlexiblePlateFirstPrinciple(POI,grids,a,b)); % model for simulation

load controller
sys = c2d(shapeit_data.P.sys,Ts); % model for ILCBF
controller = d2d(ss(shapeit_data.C_tf_z),Ts);
PS = feedback(sys,controller,-1);
%% weighting
We          = eye(N)*1e6;
Wf          = eye(N)*0e3;
WDf         = eye(N)*0e-1;
%% BF

npsi = size(Psi,2);
JPsi = zeros(N,npsi);

for iBF = 1:npsi
    JPsi(:,iBF) = lsim(PS,Psi(:,iBF));
end
R = JPsi.'*We*JPsi+Psi.'*(Wf+WDf)*Psi;
Rinv = eye(size(R,2))/R;

Q = Rinv*(JPsi.'*We*JPsi+Psi.'*WDf*Psi);
L = Rinv*(JPsi.'*We);

%% init ILC with BF
theta_jplus1 = theta0;
f_jplus1 = Psi*theta_jplus1;
%% initialize plotting and storage for ILC
PlotTrialData;

% Initialize storage variables.
history.f           = NaN(N,N_trial);
history.u           = NaN(N,N_trial);
history.e           = NaN(N,N_trial);
history.eNorm       = NaN(1,N_trial);
history.eInfNorm    = NaN(1,N_trial);
history.theta_j     = NaN(npsi,N_trial);

for trial = 1:N_trial
    f_j = f_jplus1;
    theta_j(:,trial) = theta_jplus1;
    out = sim('flexibleBeamILCBF','SrcWorkspace','current');
    
    % load simulation data:
    u_j = out.simout(:,1);
    
    e_j = out.simout(:,2);
    
    % Store trial data.
    history.f(:,trial)          = f_j;
    history.u(:,trial)          = u_j;
    history.e(:,trial)          = e_j;
    history.eNorm(:,trial)      = norm(e_j,2);
    history.eInfNorm(:,trial)   = norm(e_j,Inf);
    history.theta_j(:,trial)  = theta_j(:,trial);
    
    PlotTrialData;
    
    theta_jplus1 = (Q*theta_j(:,trial)+L*e_j);
    f_jplus1 = Psi*theta_jplus1;
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

