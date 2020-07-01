function logp = tapas_softmax_binary_controllability_2beta(r, infStates, ptrans)
% Calculates the log-probability of response y=1 under the binary softmax model
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

be_choice = exp(ptrans(1));
be_controllability = exp(ptrans(2));

% Initialize returned log-probabilities, predictions,
% and residuals as NaNs so that NaN is returned for all
% irregualar trials
n = size(infStates,1);
logp_choice = NaN(n,1);
logp_controllability = NaN(n,1);
ctr_ind = r.u(:,3);

%% Calculate log-probability of choice (not controllability choice

yhat = NaN(n,1);
res  = NaN(n,1);

yhat_controllability = NaN(n,1);
res_controllability  = NaN(n,1);

% Check input format
if size(r.u,2) ~= 1 && size(r.u,2) ~= 3
    error('tapas:hgf:SoftMaxBinary:InputsIncompatible', 'Inputs incompatible with tapas_softmax_binary observation model. See tapas_softmax_binary_config.m.')
end

% Weed irregular trials out from inferred states, responses, and inputs
x = infStates(:,1,1);
x(r.irr) = [];
y_choice = r.y(:,1);
y_choice(r.irr) = [];

if size(r.u,2) == 3
    r0 = r.u(:,2);
    r0(r.irr) = [];
    r1 = r.u(:,3);
    r1(r.irr) = [];
end

% Calculate log-probabilities for non-irregular trials
% If input matrix has only one column, assume the weight (reward value)
% of both options is equal to 1
probc = 1./(1+exp(-be_choice.*(2.*x-1).*(2.*y_choice-1)));

reg = ~ismember(1:n,r.irr);

logp_choice(reg) = log(probc);
%% Calculate log-probability of controllability choice
x_controllability = infStates(:,1,7); % controllability prediction: 5, posterior 7, Choice is based on posterior
y_controllability = r.y(:,2);

ctr_trials=find(ctr_ind);
reg_controllability = ctr_trials(~ismember(ctr_trials,r.irr));

prob_controllability = 1./(1+exp(-be_controllability.*(2.*x_controllability(reg_controllability)-1).*(2.*y_controllability(reg_controllability)-1)));
logp_controllability(reg_controllability) = log(prob_controllability);
logp_controllability(isnan(logp_controllability))=0; 
logp = logp_choice + logp_controllability;
return;
