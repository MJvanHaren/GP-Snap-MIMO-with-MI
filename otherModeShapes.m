function [Umnkl] = otherModeShapes(Xm,Yn,Xk,Yl,s,nu,D)
grids = size(Xm,1);

dd = @(sig) [0;diff(diff(sig));0];
d = @(sig) [0;diff(sig)];
Xmdd = dd(Xm);
Xmd = d(Xm);
Xkdd = dd(Xk);
Xkd = d(Xk);

Yndd = dd(Yn);
Ynd = d(Yn);
Yldd = dd(Yl);
Yld = d(Yl);

U = 0;
for x = 1:grids
    for y = 1:grids
        U = U+Xmdd(x)^2*Yn(y)^2+...
              Xkdd(x)^2*Yl(y)^2+...
              Xm(x)^2*Yndd(y)^2+... 
              Xk(x)^2*Yldd(y)^2+...
              2*nu*(Xmdd(x)*Xm(x)*Yndd(y)*Yn(y)+s*(Xm(x)*Xkdd(x)*Yndd(y)*Yl(y)+Xmdd(x)*Xk(x)*Yn(y)*Yldd(y))+Xkdd(x)*Xk(x)*Yldd(y)*Yl(y))+...
              2*(1-nu)*(Xmd(x)^2*Ynd(y)^2 + 2*s*Xmd(x)*Xkd(x)*Ynd(y)*Yld(y)+Xkd(x)^2*Yld(y)^2);
    end
end
Umnkl = U*D/2;

end