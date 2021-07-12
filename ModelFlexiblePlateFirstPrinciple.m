function [G,fn] = ModelFlexiblePlateFirstPrinciple(Cmatrix,grids,a,b)
%% porperties
                                                    
E = 2.1e11;                                                                
rho = 7850;                                                                  
nu = 0.285;
h = 2e-3;
D = E*h^3/(12*(1-nu^2));
ma = rho*h;
mass = a*b*rho*h;
%% alpha
nx = 6; ny=6;
for m = 0:1:nx
    alpha(m+1) = 0.25*(2*m-1)*pi/a;
end
for n = 0:1:ny
    beta(n+1) = 0.25*(2*n-1)*pi/b;
end
alpha(1:2) = 0;
beta(1:2) = 0;
%% natural vibrations of free beam
x = linspace(-a,a,grids)';
y = linspace(-b,b,grids)';

Xm(:,1) = ones(grids,1);
Xm(:,2) = x/a;
Yn(:,1) = ones(grids,1);
Yn(:,2) = y/b;

for m = 2:2:nx
    Xm(:,m+1) = 0.5*((cosh(alpha(m+1)*x))/(cosh(alpha(m+1)*a))+cos(alpha(m+1)*x)/(cos(alpha(m+1)*a)));
end
for m = 3:2:nx
    Xm(:,m+1) = 0.5*((sinh(alpha(m+1)*x))/(sinh(alpha(m+1)*a))+sin(alpha(m+1)*x)/(sin(alpha(m+1)*a)));
end

for n = 2:2:ny
    Yn(:,n+1) = 0.5*((cosh(beta(n+1)*y))/(cosh(beta(n+1)*b))+cos(beta(n+1)*y)./(cos(beta(n+1)*b)));
end
for n = 3:2:ny
    Yn(:,n+1) = 0.5*((sinh(beta(n+1)*y))/(sinh(beta(n+1)*b))+sin(beta(n+1)*y)./(sin(beta(n+1)*b))); 
end


for m = 2:nx
    for n = 2:nx
        u(m,n) = (alpha(m)^4+beta(n)^4)*a^2*b^2+2*nu*alpha(m)*a*beta(n)*b*tanh(alpha(m)*a)*tanh(beta(n)*b)*(1-alpha(m)*a*tanh(alpha(m)*a)*(1-beta(n)*b*tanh(beta(n)*b)+2*(1-nu)*alpha(m)*a*beta(n)*b*tanh(alpha(m)*a)*tanh(beta(n)*b)*(3+alpha(m)*a*tanh(alpha(m)*a)*(3+beta(n)*b*tanh(beta(n)*b)))));
        omega2(m,n) = (D/ma*u(m,n)/(a^2*b^2));
        Omega(m,n) = 4/pi^2*b/a*sqrt(u(m,n));
    end
end

Umnkl = @(m,n,k,l,s) otherModeShapes(Xm(:,m),Yn(:,n),Xm(:,k),Yn(:,l),s,nu,D);
omegamnkl2 = @(m,n,k,l,s) 4*Umnkl(m,n,k,l,s)/m/a/b;


omega = sqrt(abs(omega2));
%% bode calcs with Wodek

no = grids^2;
if false %MISO
    B = [[1;zeros(no-1,1)] [zeros(grids-1,1);1;zeros(no-grids,1)] [zeros(no-grids,1);1;zeros(grids-1,1)] [zeros(no-1,1);1]];
    Xnx(:,1) = reshape(Xm(:,2)*Yn(:,2)',[],1);
    Xnx(:,2) = reshape(Xm(:,2)*Yn(:,3)',[],1);
    Xnx(:,3) = reshape(Xm(:,3)*Yn(:,2)',[],1);
    Xnx(:,4) = reshape(Xm(:,3)*Yn(:,3)',[],1);
    wn = [omega(3,3) omega(3,4) omega(4,3) omega(4,4)]; 
else
    Bmatrix = [1 zeros(1,grids-2) 1;zeros(grids-2,grids);1 zeros(1,grids-2) 1];
    B = Bmatrix(:);
    Xnx(:,1) = reshape(Xm(:,3)*Yn(:,1)'+Xm(:,1)*Yn(:,3)',[],1);
    Xnx(:,2) = reshape(Xm(:,3)*Yn(:,3)',[],1);
    Xnx(:,3) = reshape(Xm(:,5)*Yn(:,3)'+Xm(:,3)*Yn(:,5)',[],1);
    wn = [omegamnkl2(3,1,1,3,1) omega(4,4) omegamnkl2(5,3,3,5,1)];
end

if false % plot mode shapes
    figure
    for i = 1:3
    [xv,yv] = meshgrid(linspace(-a,a,grids),linspace(-a,a,grids));
    thickness = 0.25/20;
    z = reshape(Xnx(:,i),grids,[])/20;
    c = z;
    % top & bottom faces
    subplot(1,3,i)
    surf(xv,yv,z+thickness,c,'edgecolor',[0 0 0],'facecolor','interp','FaceLighting','gouraud')
    hold on;
    surf(xv,yv,z-thickness,c,'edgecolor',[0 0 0],'facecolor','interp','FaceLighting','gouraud')
    % Now the 4 sides
    surf([xv(1,:); xv(1,:)],[yv(1,:); yv(1,:)], ...
            [z(1,:)+thickness; z(1,:)-thickness],[c(1,:); c(1,:)],'edgecolor',[0 0 0],'facecolor','interp','FaceLighting','gouraud')
    surf([xv(end,:); xv(end,:)],[yv(end,:); yv(end,:)], ...
            [z(end,:)+thickness; z(end,:)-thickness],[c(end,:); c(end,:)],'edgecolor',[0 0 0],'facecolor','interp','FaceLighting','gouraud')
    surf([xv(:,1), xv(:,1)],[yv(:,1), yv(:,1)], ...
            [z(:,1)+thickness, z(:,1)-thickness],[c(:,1), c(:,1)],'edgecolor',[0 0 0],'facecolor','interp','FaceLighting','gouraud')
    surf([xv(:,end), xv(:,end)],[yv(:,end), yv(:,end)], ...
            [z(:,end)+thickness, z(:,end)-thickness],[c(:,end), c(:,end)],'edgecolor',[0 0 0],'facecolor','interp','FaceLighting','gouraud');
    axis equal; grid off; axis off
    end
end

Phi = [ones(no,1)/sqrt(no) Xnx./sqrt(sum(Xnx.^2))];

fn = wn/2/pi;
C = Cmatrix(:)';

M = mass*eye(no)/no;
Mm = Phi'*M*Phi;
Bm = inv(Mm)*Phi'*B;
Cm = C*Phi;

%% iterating over modes and positions to determine Gy and Gz
s = tf('s');
G(1,1)=1/(mass*s^2);

n = length(Cm);
zeta = [0.1 0.12 0.12 0.3];
for r = 2:n
    G(1,1) = G(1,1)+(Cm(:,r)*Bm(r,:))/(s^2+2*zeta(r-1)*wn(r-1)*s+wn(r-1)^2);
end

G(2,2) = 1/(mass*s^2);
G(3,3) = 1/(mass*s^2);
end