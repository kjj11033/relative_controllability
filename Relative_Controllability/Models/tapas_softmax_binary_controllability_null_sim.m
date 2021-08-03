function [y y_ctr] = tapas_softmax_binary_controllability_null_sim(r, infStates, ptrans)
% Calculates the log-probability of response y=1 under the binary softmax model
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform zeta to its native space
% ptrans
be_choice = exp(ptrans(1));
% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
% irregualar trials
n = size(infStates,1);
ctr_ind = r.u(:,3);

%% Calculate log-probability of choice (not controllability choice

% Weed irregular trials out from inferred states, responses, and inputs
x = infStates(:,1,1);

y_choice = r.y(:,1);


%% Calculate log-probability of controllability choice
ctr_trials=find(ctr_ind);

% Apply the logistic sigmoid to the inferred states
prob = tapas_sgm(be_choice.*(2.*x-1),1);
% prob_ctr = tapas_sgm(be_controllability.*(2.*x_controllability -1),1);

% Initialize random number generator
if isnan(r.c_sim.seed)
    rng('shuffle');
else
    rng(r.c_sim.seed);
end

% Simulate
y = binornd(1, prob);
y_ctr = binornd(1, 1/2*ones(size(y)));
y_ctr(find(1-ctr_ind)) = nan;

return;
