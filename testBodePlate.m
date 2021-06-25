close all; clear all; clc;
SetPlotLatexStyle;
opts = DefBodeOpts;
%%
grids = 31;
Cmatrix = [ zeros((grids-1)/2,grids); 
            [zeros(1,(grids-1)/2) 1 zeros(1,(grids-1)/2)];
            zeros((grids-1)/2,grids)]; % center
% Cmatrix = [zeros(8,grids);
%             zeros(1,5) 1 zeros(1,25);
%             zeros(22,grids)]; % random point
% Cmatrix = [zeros(1,(grids-1)/2) 1 zeros(1,(grids-1)/2);
%             zeros(grids-1,grids)]; % edge center
        

[G,fn] = ModelFlexiblePlateFirstPrinciple(Cmatrix,grids);
figure
bodemag(G,opts);