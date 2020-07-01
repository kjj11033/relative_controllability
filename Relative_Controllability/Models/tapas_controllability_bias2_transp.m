function [pvec, pstruct] = tapas_controllability_bias2_transp(r, ptrans)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


pvec    = NaN(1,length(ptrans));
pstruct = struct;

l = 2;

pvec(1:l)         = tapas_sgm(ptrans(1:l),1);              % mu_0, % sigmoid transform
pstruct.mu_0      = pvec(1:l);
pvec(l+1:2*l)     = tapas_sgm(ptrans(l+1:2*l),1);                  % sa_0
pstruct.alpha     = pvec(l+1:2*l);
pvec(2*l+1)       = tapas_sgm(ptrans(2*l+1),1);                  % sa_0
pstruct.theta     = pvec(2*l+1);
pvec(2*l+2)       = exp(ptrans(2*l+2));                  % sa_0
pstruct.exp_pos_bias      = pvec(2*l+2);
pvec(2*l+3)       = exp(ptrans(2*l+3));                  % sa_0
pstruct.exp_neg_bias      = pvec(2*l+3);


return;
